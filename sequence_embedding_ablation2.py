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
OUT_DIR =  "/media/experiment/ssd_1TB/multiomic-causal/data/processed"
CACHE_PATH = os.path.join(OUT_DIR, "swspm_seq_features_cache.pt")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# FFT-based SWSPM (Sliding Window Signal Processing Method) hyperparameters
SWSPM_N_FFT = 128          # Fixed zero-padded FFT window length -> 65 one-sided spectral bins per channel
SWSPM_WIN_LEN = 64         # Sliding window length along sequence
SWSPM_STEP = 16            # Sliding window stride

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


def _sliding_window_fft_spectrum(signal_1d, win_len=SWSPM_WIN_LEN, step=SWSPM_STEP, n_fft=SWSPM_N_FFT):
    """Compute mean and max normalized power spectral density across sliding windows via rFFT."""
    L = len(signal_1d)
    if L == 0:
        return np.zeros((n_fft // 2 + 1) * 2, dtype=np.float32)

    # Extract overlapping sliding windows (pad short sequences to win_len)
    if L <= win_len:
        windows = [np.pad(signal_1d, (0, win_len - L), mode="constant")]
    else:
        windows = [signal_1d[start:start + win_len] for start in range(0, L - win_len + 1, step)]
        if (L - win_len) % step != 0:
            windows.append(signal_1d[-win_len:])

    win_mat = np.stack(windows, axis=0).astype(np.float32)  # (num_windows, win_len)
    # Remove DC offset per window and apply Hann taper before zero-padded rFFT
    win_mat = win_mat - win_mat.mean(axis=1, keepdims=True)
    hann = np.hanning(win_len).astype(np.float32)[None, :]
    spec = np.abs(np.fft.rfft(win_mat * hann, n=n_fft, axis=1))  # (num_windows, n_fft//2 + 1)

    # Normalize each window spectrum to unit L2 energy to make features length-invariant
    spec = spec / (np.linalg.norm(spec, axis=1, keepdims=True) + 1e-6)
    spec_mean = spec.mean(axis=0)
    spec_max = spec.max(axis=0)
    return np.concatenate([spec_mean, spec_max]).astype(np.float32)


def rna_swspm_features(seq):
    """
    FFT-based SWSPM embedding for RNA sequences:
      - Channel 1: Electron-Ion Interaction Potential (EIIP: A=0.1260, C=0.1340, G=0.0806, U/T=0.1335)
      - Channel 2: Purine/Pyrimidine (R/Y) structural indicator (+1 for A/G, -1 for C/U)
      - Channel 3: Strong/Weak H-bond (S/W) indicator (+1 for G/C, -1 for A/U)
    Each channel yields 2 * (SWSPM_N_FFT // 2 + 1) = 130 spectral features -> 390D total.
    """
    bases = "ACGT"
    seq = "".join([c for c in seq.upper().replace("U", "T") if c in bases])
    if len(seq) < 10:
        return np.zeros(390, dtype=np.float32)

    eiip_map = {"A": 0.1260, "C": 0.1340, "G": 0.0806, "T": 0.1335}
    ry_map = {"A": 1.0, "G": 1.0, "C": -1.0, "T": -1.0}
    sw_map = {"G": 1.0, "C": 1.0, "A": -1.0, "T": -1.0}

    sig_eiip = np.array([eiip_map[c] for c in seq], dtype=np.float32)
    sig_ry = np.array([ry_map[c] for c in seq], dtype=np.float32)
    sig_sw = np.array([sw_map[c] for c in seq], dtype=np.float32)

    return np.concatenate([
        _sliding_window_fft_spectrum(sig_eiip),
        _sliding_window_fft_spectrum(sig_ry),
        _sliding_window_fft_spectrum(sig_sw),
    ])


def prot_swspm_features(seq):
    """
    FFT-based SWSPM embedding for TF/RBP protein sequences:
      - Channel 1: Electron-Ion Interaction Potential (EIIP) per amino acid
      - Channel 2: Kyte-Doolittle Hydropathy index
      - Channel 3: Net electrostatic charge / polarity signal
    Each channel yields 2 * (SWSPM_N_FFT // 2 + 1) = 130 spectral features -> 390D total.
    """
    aas = "ACDEFGHIKLMNPQRSTVWY"
    seq = "".join([c for c in seq.upper() if c in aas])
    if len(seq) < 5:
        return np.zeros(390, dtype=np.float32)

    eiip_aa = {
        "A": 0.0373, "C": 0.0829, "D": 0.1263, "E": 0.0058, "F": 0.0946,
        "G": 0.0050, "H": 0.0242, "I": 0.0000, "K": 0.0371, "L": 0.0000,
        "M": 0.0823, "N": 0.0036, "P": 0.0198, "Q": 0.0761, "R": 0.0959,
        "S": 0.0829, "T": 0.0941, "V": 0.0057, "W": 0.0548, "Y": 0.0516,
    }
    kd_hydro = {
        "A": 1.8, "C": 2.5, "D": -3.5, "E": -3.5, "F": 2.8,
        "G": -0.4, "H": -3.2, "I": 4.5, "K": -3.9, "L": 3.8,
        "M": 1.9, "N": -3.5, "P": -1.6, "Q": -3.5, "R": -4.5,
        "S": -0.8, "T": -0.7, "V": 4.2, "W": -0.9, "Y": -1.3,
    }
    charge_aa = {
        "D": -1.0, "E": -1.0, "K": 1.0, "R": 1.0, "H": 0.5,
        "A": 0.0, "C": 0.0, "F": 0.0, "G": 0.0, "I": 0.0,
        "L": 0.0, "M": 0.0, "N": 0.0, "P": 0.0, "Q": 0.0,
        "S": 0.0, "T": 0.0, "V": 0.0, "W": 0.0, "Y": 0.0,
    }

    sig_eiip = np.array([eiip_aa[c] for c in seq], dtype=np.float32)
    sig_kd = np.array([kd_hydro[c] for c in seq], dtype=np.float32)
    sig_chg = np.array([charge_aa[c] for c in seq], dtype=np.float32)

    return np.concatenate([
        _sliding_window_fft_spectrum(sig_eiip),
        _sliding_window_fft_spectrum(sig_kd),
        _sliding_window_fft_spectrum(sig_chg),
    ])


def load_or_build_sequence_features():
    if os.path.exists(CACHE_PATH):
        print("Loading cached FFT-SWSPM sequence embeddings from:", CACHE_PATH)
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

    lnc_seq_raw = np.zeros((n_lnc, 390), dtype=np.float32)
    for i, sym in enumerate(lnc_list):
        enst = sym_to_enst.get(sym)
        if enst and enst in enst_seqs:
            lnc_seq_raw[i] = rna_swspm_features(enst_seqs[enst])

    prot_seq_raw = np.zeros((n_prot, 390), dtype=np.float32)
    for i, sym in enumerate(prot_list):
        ensp = sym_to_ensp.get(sym)
        if ensp and ensp in ensp_seqs:
            prot_seq_raw[i] = prot_swspm_features(ensp_seqs[ensp])

    # Standardize raw FFT-SWSPM sequence matrices
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
    run_seq_ablation("+ Raw FFT-SWSPM Sequence (390D)", mode="raw"),
    run_seq_ablation("+ SVD-Compressed FFT-SWSPM (16D)", mode="svd"),
])
df_seq.to_csv(os.path.join(OUT_DIR, "swspm_sequence_ablation_results.csv"), index=False)
print("\n=== FFT-SWSPM SEQUENCE EMBEDDING ABLATION RESULTS (HELD-OUT TEST SET) ===")
print(df_seq.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
