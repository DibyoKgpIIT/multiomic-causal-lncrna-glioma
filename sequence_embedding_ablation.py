import os
import json
import time
import urllib.request
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUT_DIR = os.path.join(BASE_DIR, "data/processed")
CACHE_PATH = os.path.join(OUT_DIR, "seq_features_cache.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Set True if you want to use HuggingFace ESM-2 (protein) + RNA-FM/RiNALMo instead of fast k-mer/biophysical seq features
USE_FOUNDATION_LM = False

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
base_data = bundle["graph"]
node_maps = bundle["node_maps"]
splits = torch.load(os.path.join(OUT_DIR, "multitask_splits.pt"), weights_only=False)

for et in list(base_data.edge_types):
    if base_data[et].edge_index.numel() == 0:
        del base_data[et]
    elif "edge_attr" in base_data[et]:
        del base_data[et].edge_attr

lnc_list = sorted(node_maps["lncrna"], key=node_maps["lncrna"].get)
prot_list = sorted(node_maps["protein"], key=node_maps["protein"].get)
n_lnc, n_prot = len(lnc_list), len(prot_list)
valid_lnc_mask = base_data["lncrna"].strict_lnc_mask


def fetch_mygene_sequences(symbols, fields="symbol,ensembl.transcript,refseq.rna,uniprot.Swiss-Prot"):
    """Batch query MyGene.info for transcript/protein accessions."""
    res_map = {}
    url = "https://mygene.info/v3/gene"
    for start in range(0, len(symbols), 1000):
        chunk = symbols[start:start + 1000]
        payload = json.dumps({"ids": chunk, "scopes": "symbol,alias", "species": "human", "fields": fields}).encode("utf-8")
        req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                hits = json.loads(resp.read().decode("utf-8"))
            for h in hits:
                q = h.get("query")
                if q and q not in res_map and not h.get("notfound"):
                    res_map[q] = h
        except Exception as e:
            print(f"Batch {start} warning: {e}")
        time.sleep(0.2)
    return res_map


def fetch_ensembl_seqs(enst_ids, seq_type="cdna"):
    """Batch fetch FASTA sequences from Ensembl REST API (up to 50 IDs per POST)."""
    seq_map = {}
    url = f"https://rest.ensembl.org/sequence/id?type={seq_type}"
    unique_ids = list(dict.fromkeys([x for x in enst_ids if x]))
    for start in range(0, len(unique_ids), 50):
        chunk = unique_ids[start:start + 50]
        payload = json.dumps({"ids": chunk}).encode("utf-8")
        req = urllib.request.Request(
            url, data=payload, headers={"Content-Type": "application/json", "Accept": "application/json"}
        )
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                items = json.loads(resp.read().decode("utf-8"))
            for item in items:
                qid = item.get("query") or item.get("id")
                if qid and item.get("seq"):
                    seq_map[qid] = item["seq"].upper()
        except Exception:
            pass
        time.sleep(0.15)
    return seq_map


def rna_kmer_features(seq):
    """Compute 3-mer (64D) + 4-mer (256D) + GC/CpG/AU-rich biophysical stats (4D) = 324D."""
    bases = "ACGT"
    seq = "".join([c for c in seq.upper().replace("U", "T") if c in bases])
    if len(seq) < 10:
        return np.zeros(324, dtype=np.float32)

    k3_idx = {a + b + c: i for i, a in enumerate(bases) for b in bases for c in bases}
    # Fix indexing cleanly
    k3_list = [a + b + c for a in bases for b in bases for c in bases]
    k3_map = {k: i for i, k in enumerate(k3_list)}
    k4_list = [a + b + c + d for a in bases for b in bases for c in bases for d in bases]
    k4_map = {k: i for i, k in enumerate(k4_list)}

    v3 = np.zeros(64, dtype=np.float32)
    for i in range(len(seq) - 2):
        v3[k3_map[seq[i:i + 3]]] += 1.0
    v3 /= (v3.sum() + 1e-6)

    v4 = np.zeros(256, dtype=np.float32)
    for i in range(len(seq) - 3):
        v4[k4_map[seq[i:i + 4]]] += 1.0
    v4 /= (v4.sum() + 1e-6)

    gc = (seq.count("G") + seq.count("C")) / len(seq)
    cpg = seq.count("CG") / max(1, len(seq) - 1)
    are = seq.count("ATTTA") / max(1, len(seq) - 4)  # AU-rich stability element
    log_len = np.log1p(len(seq)) / 10.0
    bio = np.array([gc, cpg, are, log_len], dtype=np.float32)
    return np.concatenate([v3, v4, bio])


def prot_seq_features(seq):
    """Compute Amino Acid Composition (20D) + Dipeptide Composition (400D) = 420D."""
    aas = "ACDEFGHIKLMNPQRSTVWY"
    seq = "".join([c for c in seq.upper() if c in aas])
    if len(seq) < 5:
        return np.zeros(420, dtype=np.float32)
    aa_map = {a: i for i, a in enumerate(aas)}
    dp_list = [a + b for a in aas for b in aas]
    dp_map = {d: i for i, d in enumerate(dp_list)}

    v1 = np.zeros(20, dtype=np.float32)
    for c in seq:
        v1[aa_map[c]] += 1.0
    v1 /= (v1.sum() + 1e-6)

    v2 = np.zeros(400, dtype=np.float32)
    for i in range(len(seq) - 1):
        v2[dp_map[seq[i:i + 2]]] += 1.0
    v2 /= (v2.sum() + 1e-6)
    return np.concatenate([v1, v2])


def load_or_build_sequence_features():
    if os.path.exists(CACHE_PATH):
        print("Loading cached sequence embeddings from:", CACHE_PATH)
        return torch.load(CACHE_PATH, weights_only=False)

    print("Fetching transcript & protein sequences for HGNC-verified lncRNAs and TFs/Proteins...")
    active_lncs = [lnc_list[i] for i in range(n_lnc) if valid_lnc_mask[i]]
    lnc_meta = fetch_mygene_sequences(active_lncs, fields="symbol,ensembl.transcript")

    sym_to_enst = {}
    for sym, info in lnc_meta.items():
        ens = info.get("ensembl", {})
        if isinstance(ens, list):
            ens = ens[0]
        txs = ens.get("transcript", [])
        if isinstance(txs, str):
            txs = [txs]
        if txs:
            sym_to_enst[sym] = txs[0]

    enst_seqs = fetch_ensembl_seqs(list(sym_to_enst.values()), seq_type="cdna")
    print(f"Retrieved cDNA sequences for {len(enst_seqs)} / {len(active_lncs)} verified lncRNAs")

    # Fetch protein sequences for CollecTRI TFs + top interacting proteins
    tf_ei = base_data["protein", "regulates", "mrna"].edge_index.numpy()
    bind_ei = base_data["lncrna", "binds", "protein"].edge_index.numpy()
    active_prot_idx = sorted(set(tf_ei[0].tolist()) | set(bind_ei[1].tolist()))
    active_prots = [prot_list[i] for i in active_prot_idx]

    prot_meta = fetch_mygene_sequences(active_prots, fields="symbol,ensembl.protein")
    sym_to_ensp = {}
    for sym, info in prot_meta.items():
        ens = info.get("ensembl", {})
        if isinstance(ens, list):
            ens = ens[0]
        ps = ens.get("protein", [])
        if isinstance(ps, str):
            ps = [ps]
        if ps:
            sym_to_ensp[sym] = ps[0]

    ensp_seqs = fetch_ensembl_seqs(list(sym_to_ensp.values()), seq_type="protein")
    print(f"Retrieved peptide sequences for {len(ensp_seqs)} / {len(active_prots)} active TFs/RBPs")

    lnc_seq_raw = np.zeros((n_lnc, 324), dtype=np.float32)
    for i, sym in enumerate(lnc_list):
        enst = sym_to_enst.get(sym)
        if enst and enst in enst_seqs:
            lnc_seq_raw[i] = rna_kmer_features(enst_seqs[enst])

    prot_seq_raw = np.zeros((n_prot, 420), dtype=np.float32)
    for i, sym in enumerate(prot_list):
        ensp = sym_to_ensp.get(sym)
        if ensp and ensp in ensp_seqs:
            prot_seq_raw[i] = prot_seq_features(ensp_seqs[ensp])

    # Standardize raw sequence matrices
    lnc_seq_std = (lnc_seq_raw - lnc_seq_raw.mean(0, keepdims=True)) / (lnc_seq_raw.std(0, keepdims=True) + 1e-6)
    prot_seq_std = (prot_seq_raw - prot_seq_raw.mean(0, keepdims=True)) / (prot_seq_raw.std(0, keepdims=True) + 1e-6)

    # Also build 16D SVD-compressed versions
    lnc_svd = TruncatedSVD(n_components=16, random_state=42).fit_transform(lnc_seq_std)
    lnc_svd = (lnc_svd - lnc_svd.mean(0, keepdims=True)) / (lnc_svd.std(0, keepdims=True) + 1e-6)

    prot_svd = TruncatedSVD(n_components=16, random_state=42).fit_transform(prot_seq_std)
    prot_svd = (prot_svd - prot_svd.mean(0, keepdims=True)) / (prot_svd.std(0, keepdims=True) + 1e-6)

    cache = {
        "lnc_raw": torch.tensor(lnc_seq_std, dtype=torch.float32),
        "prot_raw": torch.tensor(prot_seq_std, dtype=torch.float32),
        "lnc_svd": torch.tensor(lnc_svd, dtype=torch.float32),
        "prot_svd": torch.tensor(prot_svd, dtype=torch.float32),
    }
    torch.save(cache, CACHE_PATH)
    return cache


seq_cache = load_or_build_sequence_features()

# Prepare exact same entity-disjoint split as train_unsigned_causal.py
y_drv = splits["task2"]["y_driver"].clone()
y_dir = splits["task2"]["y_dir"].clone()
y_drv[~valid_lnc_mask] = -1.0

orig_train = splits["task2"]["train_idx"][y_drv[splits["task2"]["train_idx"]] >= 0]
val_idx = splits["task2"]["val_idx"][y_drv[splits["task2"]["val_idx"]] >= 0]
test_idx = splits["task2"]["test_idx"][y_drv[splits["task2"]["test_idx"]] >= 0]

rng = np.random.default_rng(42)
perm_tr = rng.permutation(len(orig_train))
cut = int(0.85 * len(orig_train))
inner_tr_idx = orig_train[perm_tr[:cut]]
inner_va_idx = orig_train[perm_tr[cut:]]

heldout_lncs = set(inner_va_idx.tolist()) | set(val_idx.tolist()) | set(test_idx.tolist())
t1_tr_ei, t1_tr_y, t1_tr_w = splits["task1"]["train"]
keep_t1_tr = torch.tensor([int(u) not in heldout_lncs for u in t1_tr_ei[0]], dtype=torch.bool)
t1_tr_ei = t1_tr_ei[:, keep_t1_tr].to(device)
t1_tr_y = t1_tr_y[keep_t1_tr].to(device)
t1_tr_w = t1_tr_w[keep_t1_tr].to(device)

t1_va_ei, t1_va_y, t1_va_w = [x.to(device) for x in splits["task1"]["val"]]
t1_te_ei, t1_te_y, t1_te_w = [x.to(device) for x in splits["task1"]["test"]]

y_drv_d = y_drv.to(device)
y_dir_d = y_dir.to(device)
inner_tr_d = inner_tr_idx.to(device)
inner_va_d = inner_va_idx.to(device)
val_d = val_idx.to(device)
test_d = test_idx.to(device)

pos_weight_drv = torch.tensor([(y_drv_d[inner_tr_d] == 0).sum() / (y_drv_d[inner_tr_d] == 1).sum()], device=device)
pol_tr = inner_tr_d[y_dir_d[inner_tr_d] >= 0]
pos_weight_pol = torch.tensor([(y_dir_d[pol_tr] == 0).sum() / (y_dir_d[pol_tr] == 1).sum()], device=device)


class UnsignedMultiOmicCausalGNN(nn.Module):
    def __init__(self, data_obj, hidden_dim=64, dropout=0.35):
        super().__init__()
        self.dropout = dropout
        self.feat_proj = nn.ModuleDict({
            nt: Linear(data_obj[nt].x.size(1), hidden_dim)
            for nt in ["lncrna", "mrna", "protein"]
        })
        self.id_emb = nn.ModuleDict({
            nt: nn.Embedding(data_obj[nt].num_nodes, hidden_dim)
            for nt in ["mirna", "disease", "drug"]
            if nt in data_obj.node_types and data_obj[nt].num_nodes > 0
        })
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(2):
            conv_dict = {et: SAGEConv((-1, -1), hidden_dim) for et in data_obj.edge_types}
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in data_obj.node_types}))

        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim, 1)
        )
        self.driver_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )
        self.polarity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden_dim // 2, 1)
        )

    def encode_nodes(self, data_obj):
        h = {}
        for nt in data_obj.node_types:
            if nt in self.feat_proj:
                h[nt] = F.gelu(self.feat_proj[nt](data_obj[nt].x))
            elif nt in self.id_emb:
                h[nt] = self.id_emb[nt](data_obj[nt].node_id)
        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, data_obj.edge_index_dict)
            for nt in h:
                if nt in h_next:
                    h[nt] = norm[nt](h[nt] + F.dropout(F.gelu(h_next[nt]), p=self.dropout, training=self.training))
        return h

    def decode_links(self, h_lnc, h_dis, edge_index):
        src = h_lnc[edge_index[0]]
        dst = h_dis[edge_index[1]]
        return self.link_head(torch.cat([src, dst, src * dst], dim=-1)).squeeze(-1)


def run_seq_ablation(label, mode="none"):
    torch.manual_seed(42)
    g = base_data.clone()
    if mode == "raw":
        g["lncrna"].x = torch.cat([g["lncrna"].x, seq_cache["lnc_raw"]], dim=1)
        g["protein"].x = torch.cat([g["protein"].x, seq_cache["prot_raw"]], dim=1)
    elif mode == "svd":
        g["lncrna"].x = torch.cat([g["lncrna"].x, seq_cache["lnc_svd"]], dim=1)
        g["protein"].x = torch.cat([g["protein"].x, seq_cache["prot_svd"]], dim=1)
    g = g.to(device)

    model = UnsignedMultiOmicCausalGNN(g, hidden_dim=64, dropout=0.35).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-3)

    best_hmean = -1.0
    best_ep = 0
    best_state = None

    for ep in range(1, 41):
        model.train()
        opt.zero_grad()
        h = model.encode_nodes(g)
        loss_t1 = F.binary_cross_entropy_with_logits(
            model.decode_links(h["lncrna"], h["disease"], t1_tr_ei), t1_tr_y, weight=t1_tr_w
        )
        loss_drv = F.binary_cross_entropy_with_logits(
            model.driver_head(h["lncrna"][inner_tr_d]).squeeze(-1), y_drv_d[inner_tr_d], pos_weight=pos_weight_drv
        )
        loss_pol = F.binary_cross_entropy_with_logits(
            model.polarity_head(h["lncrna"][pol_tr]).squeeze(-1), y_dir_d[pol_tr], pos_weight=pos_weight_pol
        )
        loss = 0.6 * loss_t1 + 1.4 * loss_drv + 0.5 * loss_pol
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            he = model.encode_nodes(g)
            p1 = torch.sigmoid(model.decode_links(he["lncrna"], he["disease"], t1_va_ei)).cpu().numpy()
            y1, w1 = t1_va_y.cpu().numpy(), t1_va_w.cpu().numpy()
            va_hard = roc_auc_score(y1[w1 == 1.0], p1[w1 == 1.0])
            p2 = torch.sigmoid(model.driver_head(he["lncrna"][inner_va_d]).squeeze(-1)).cpu().numpy()
            va_drv = roc_auc_score(y_drv_d[inner_va_d].cpu().numpy(), p2)
            hmean = 2 * va_hard * va_drv / (va_hard + va_drv + 1e-8)
            if hmean > best_hmean:
                best_hmean = hmean
                best_ep = ep
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        ht = model.encode_nodes(g)
        p1 = torch.sigmoid(model.decode_links(ht["lncrna"], ht["disease"], t1_te_ei)).cpu().numpy()
        y1, w1 = t1_te_y.cpu().numpy(), t1_te_w.cpu().numpy()
        te_link = roc_auc_score(y1, p1)
        te_hard = roc_auc_score(y1[w1 == 1.0], p1[w1 == 1.0])
        p2 = torch.sigmoid(model.driver_head(ht["lncrna"][test_d]).squeeze(-1)).cpu().numpy()
        y2 = y_drv_d[test_d].cpu().numpy()
        te_drv = roc_auc_score(y2, p2)
        te_ap = average_precision_score(y2, p2)

    return {
        "Variant": label,
        "Best_Ep": best_ep,
        "Test_Link_AUROC": te_link,
        "Test_Hard_Causal_AUROC": te_hard,
        "Test_Glioma_Driver_AUROC": te_drv,
        "Test_Glioma_Driver_AUPRC": te_ap,
    }


df_seq = pd.DataFrame([
    run_seq_ablation("Baseline Multi-Omic Only (d=64)", mode="none"),
    run_seq_ablation("+ Raw RNA & TF Sequence (324D/420D)", mode="raw"),
    run_seq_ablation("+ SVD-Compressed Sequence (16D)", mode="svd"),
])
df_seq.to_csv(os.path.join(OUT_DIR, "sequence_ablation_results.csv"), index=False)
print("\n=== SEQUENCE EMBEDDING ABLATION RESULTS (HELD-OUT TEST SET) ===")
print(df_seq.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

