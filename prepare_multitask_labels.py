import os
import re
import torch
import numpy as np
import pandas as pd

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
COMB_DIR = os.path.join(BASE_DIR, "data/lncRNA_combined")
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUT_DIR = os.path.join(BASE_DIR, "data/processed")

def norm_sym(s):
    return str(s).strip().upper() if pd.notna(s) else ""

def norm_dis(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()) if pd.notna(s) else ""

# 1. Load graph & repair drug nodes (supports Symbol, Entrez, and UniProt)
bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]

chg = pd.read_csv(os.path.join(COMB_DIR, "ChG-Miner_miner-chem-gene.tsv"), sep="\t")
chg.columns = ["drug", "gene_id"]

hgnc = pd.read_csv(
    os.path.join(RAW_DIR, "hgnc_complete_set.txt"),
    sep="\t",
    usecols=["symbol", "entrez_id", "uniprot_ids"],
    low_memory=False
)
id_to_sym = {}
for _, r in hgnc.iterrows():
    sym = norm_sym(r["symbol"])
    if not sym:
        continue
    id_to_sym[sym] = sym
    if pd.notna(r["entrez_id"]):
        id_to_sym[str(int(r["entrez_id"]))] = sym
    if pd.notna(r["uniprot_ids"]):
        for uid in str(r["uniprot_ids"]).split("|"):
            id_to_sym[uid.strip().upper()] = sym

chg["gene_sym"] = chg["gene_id"].astype(str).str.strip().str.upper().map(id_to_sym)
chg = chg.dropna(subset=["drug", "gene_sym"])
chg = chg[chg["gene_sym"].isin(node_maps["protein"])].copy()

drug_list = sorted(chg["drug"].unique())
node_maps["drug"] = {d: i for i, d in enumerate(drug_list)}
data["drug"].num_nodes = len(drug_list)
data["drug"].node_id = torch.arange(len(drug_list), dtype=torch.long)

dp_edges = sorted({(node_maps["drug"][r["drug"]], node_maps["protein"][r["gene_sym"]]) for _, r in chg.iterrows()})
if dp_edges:
    dp_ei = torch.tensor(dp_edges, dtype=torch.long).t().contiguous()
    data["drug", "targets", "protein"].edge_index = dp_ei
    data["protein", "rev_targets", "drug"].edge_index = dp_ei.flip(0)

torch.save({"graph": data, "node_maps": node_maps}, os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"))
print(f"Drugs populated: {len(drug_list)} nodes, {len(dp_edges)} drug-protein edges")

# ==========================================
# 2. Task 1 Labels: Pan-Disease Causality
# ==========================================
df_ld = pd.read_csv(os.path.join(COMB_DIR, "all_ncRNA_disease_information_LncRNADisease3.tsv"), sep="\t", low_memory=False)
df_ld = df_ld[(df_ld["Species"] == "Homo sapiens") & (df_ld["ncRNA Category"].str.lower() == "lncrna")].copy()
df_ld["lnc_sym"] = df_ld["ncRNA Symbol"].map(norm_sym)
df_ld["dis_norm"] = df_ld["Disease Name"].map(norm_dis)

df_ld = df_ld[df_ld["lnc_sym"].isin(node_maps["lncrna"]) & df_ld["dis_norm"].isin(node_maps["disease"])].copy()

# If a pair has Causality == 'Yes' in any paper, treat as causal positive (1); if only 'No', hard negative (0)
pair_labels = {}
for _, r in df_ld.iterrows():
    u = node_maps["lncrna"][r["lnc_sym"]]
    v = node_maps["disease"][r["dis_norm"]]
    is_causal = 1 if str(r["Causality"]).strip().lower() == "yes" else 0
    pair_labels[(u, v)] = max(pair_labels.get((u, v), 0), is_causal)

pos_pairs = [p for p, y in pair_labels.items() if y == 1]
hard_neg_pairs = [p for p, y in pair_labels.items() if y == 0]

# Add 2x random unobserved negatives per causal positive
rng = np.random.default_rng(42)
n_lnc = data["lncrna"].num_nodes
n_dis = data["disease"].num_nodes
all_known = set(pair_labels.keys())
rand_negs = set()
while len(rand_negs) < len(pos_pairs) * 2:
    u = int(rng.integers(0, n_lnc))
    v = int(rng.integers(0, n_dis))
    if (u, v) not in all_known:
        rand_negs.add((u, v))

# Combine into edge_label_index, edge_label, edge_weight
all_edges = pos_pairs + hard_neg_pairs + sorted(rand_negs)
all_y = [1.0] * len(pos_pairs) + [0.0] * len(hard_neg_pairs) + [0.0] * len(rand_negs)
all_w = [1.0] * len(pos_pairs) + [1.0] * len(hard_neg_pairs) + [0.5] * len(rand_negs)

perm = rng.permutation(len(all_edges))
edges_t = torch.tensor([all_edges[i] for i in perm], dtype=torch.long).t().contiguous()
y_t = torch.tensor([all_y[i] for i in perm], dtype=torch.float32)
w_t = torch.tensor([all_w[i] for i in perm], dtype=torch.float32)

n_tot = len(perm)
n_train, n_val = int(0.7 * n_tot), int(0.15 * n_tot)
task1_splits = {
    "train": (edges_t[:, :n_train], y_t[:n_train], w_t[:n_train]),
    "val": (edges_t[:, n_train:n_train+n_val], y_t[n_train:n_train+n_val], w_t[n_train:n_train+n_val]),
    "test": (edges_t[:, n_train+n_val:], y_t[n_train+n_val:], w_t[n_train+n_val:]),
}
print(f"Task 1 pairs -> Causal Pos: {len(pos_pairs)}, Hard Neg (Causality=No): {len(hard_neg_pairs)}, Rand Neg: {len(rand_negs)}")

# ==========================================
# 3. Task 2 Labels: Glioma Causal Driver & Polarity
# ==========================================
df_wiki = pd.read_csv(os.path.join(COMB_DIR, "LncRNAWiki_glioma.csv"), on_bad_lines="skip")
df_wiki["lnc_sym"] = df_wiki["symbol"].map(norm_sym)
df_wiki = df_wiki[df_wiki["lnc_sym"].isin(node_maps["lncrna"])].copy()

glioma_ld_causal = set(
    df_ld[(df_ld["Causality"].str.lower() == "yes") & (df_ld["dis_norm"].str.contains("glioma|glioblastoma"))]["lnc_sym"]
)
glioma_wiki_syms = set(df_wiki["lnc_sym"])
glioma_pos_syms = glioma_ld_causal | glioma_wiki_syms

# Clean negatives: lncRNAs causal for OTHER diseases in LncRNADisease3, excluding any glioma mention
other_causal_syms = set(df_ld[df_ld["Causality"].str.lower() == "yes"]["lnc_sym"]) - glioma_pos_syms

y_driver = torch.full((n_lnc,), -1.0, dtype=torch.float32)
for s in glioma_pos_syms:
    y_driver[node_maps["lncrna"][s]] = 1.0
for s in other_causal_syms:
    y_driver[node_maps["lncrna"][s]] = 0.0

# Polarity (1 = Up-regulated / Oncogenic, 0 = Down-regulated / Tumor-suppressive)
y_dir = torch.full((n_lnc,), -1.0, dtype=torch.float32)
for s, grp in df_wiki.groupby("lnc_sym"):
    vals = grp["expression_detail"].dropna().str.lower().tolist()
    up_cnt = sum("up" in v for v in vals)
    dn_cnt = sum("down" in v for v in vals)
    if up_cnt > dn_cnt:
        y_dir[node_maps["lncrna"][s]] = 1.0
    elif dn_cnt > up_cnt:
        y_dir[node_maps["lncrna"][s]] = 0.0

labeled_driver_idx = torch.where(y_driver >= 0)[0].numpy()
rng.shuffle(labeled_driver_idx)
nd = len(labeled_driver_idx)
nd_tr, nd_va = int(0.7 * nd), int(0.15 * nd)

task2_splits = {
    "y_driver": y_driver,
    "y_dir": y_dir,
    "train_idx": torch.tensor(labeled_driver_idx[:nd_tr], dtype=torch.long),
    "val_idx": torch.tensor(labeled_driver_idx[nd_tr:nd_tr+nd_va], dtype=torch.long),
    "test_idx": torch.tensor(labeled_driver_idx[nd_tr+nd_va:], dtype=torch.long),
}
print(f"Task 2 Glioma Drivers -> Pos: {int((y_driver == 1).sum())}, Non-Glioma Causal Neg: {int((y_driver == 0).sum())}, Polarity Labeled: {int((y_dir >= 0).sum())}")

torch.save({"task1": task1_splits, "task2": task2_splits}, os.path.join(OUT_DIR, "multitask_splits.pt"))
