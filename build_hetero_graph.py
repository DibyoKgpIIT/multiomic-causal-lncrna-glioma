import os
import re
import torch
import numpy as np
import pandas as pd
from torch_geometric.data import HeteroData
import torch_geometric.transforms as T

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
XENA_DIR = os.path.join(BASE_DIR, "data/xena_toil")
COMB_DIR = os.path.join(BASE_DIR, "data/lncRNA_combined")
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUT_DIR = os.path.join(BASE_DIR, "data/processed")
os.makedirs(OUT_DIR, exist_ok=True)

def norm_sym(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"^has-mir-", "hsa-miR-", s, flags=re.IGNORECASE)
    s = re.sub(r"^hsa-mir-", "hsa-miR-", s, flags=re.IGNORECASE)
    return s.upper()

# ==========================================
# 1. Load Probe Map & Xena TOIL Expression
# ==========================================
probemap = pd.read_csv(os.path.join(XENA_DIR, "gencode.v23.annotation.gene.probemap"), sep="\t")
probemap["ensembl_id"] = probemap["id"].str.split(".").str[0]
probemap = probemap[~probemap["id"].str.contains("_PAR_Y", na=False)].copy()
ensg_to_sym = dict(zip(probemap["ensembl_id"], probemap["gene"].map(norm_sym)))

df_expr = pd.read_parquet(os.path.join(XENA_DIR, "glioma_gtex_log2tpm1.parquet"))
pheno = pd.read_csv(os.path.join(XENA_DIR, "glioma_gtex_phenotype.csv"))

df_expr["symbol"] = df_expr["ensembl_id"].map(ensg_to_sym)
df_expr = df_expr.dropna(subset=["symbol"])
df_expr = df_expr[df_expr["symbol"] != ""].copy()

gtex_cols = [c for c in pheno.loc[pheno["_study"] == "GTEX", "sample"] if c in df_expr.columns]
lgg_cols = [
    c for c in pheno.loc[
        (pheno["_study"] == "TCGA") & (pheno["primary disease or tissue"].str.lower() == "brain lower grade glioma"),
        "sample"
    ] if c in df_expr.columns
]
gbm_cols = [
    c for c in pheno.loc[
        (pheno["_study"] == "TCGA") & (pheno["primary disease or tissue"].str.lower() == "glioblastoma multiforme"),
        "sample"
    ] if c in df_expr.columns
]

all_sample_cols = gtex_cols + lgg_cols + gbm_cols
df_gene_expr = df_expr.groupby("symbol")[all_sample_cols].mean()
expr_symbols = set(df_gene_expr.index)

# ==========================================
# 2. Load RNAInter (RR & RP) + CollecTRI
# ==========================================
df_rr = pd.read_csv(os.path.join(COMB_DIR, "Download_data_RR.txt"), sep="\t", low_memory=False)
df_rr = df_rr[(df_rr["Species1"] == "Homo sapiens") & (df_rr["Species2"] == "Homo sapiens")].copy()
df_rr["sym1"] = df_rr["Interactor1.Symbol"].map(norm_sym)
df_rr["sym2"] = df_rr["Interactor2.Symbol"].map(norm_sym)
df_rr["cat1"] = df_rr["Category1"].str.lower().str.strip()
df_rr["cat2"] = df_rr["Category2"].str.lower().str.strip()

valid_rna = {"lncrna", "mrna", "mirna"}
df_rr = df_rr[
    df_rr["cat1"].isin(valid_rna) & df_rr["cat2"].isin(valid_rna) &
    (df_rr["sym1"] != "") & (df_rr["sym2"] != "") & (df_rr["sym1"] != df_rr["sym2"])
].copy()

df_rp = pd.read_csv(os.path.join(COMB_DIR, "Download_data_RP.txt"), sep="\t", low_memory=False)
df_rp = df_rp[(df_rp["Species1"] == "Homo sapiens") & (df_rp["Species2"] == "Homo sapiens")].copy()
df_rp["sym1"] = df_rp["Interactor1.Symbol"].map(norm_sym)
df_rp["sym2"] = df_rp["Interactor2.Symbol"].map(norm_sym)
df_rp["cat1"] = df_rp["Category1"].str.lower().str.strip()
df_rp = df_rp[
    df_rp["cat1"].isin(valid_rna) & (df_rp["sym1"] != "") & (df_rp["sym2"] != "")
].copy()

df_tf = pd.read_csv(os.path.join(RAW_DIR, "collectri_human.csv"))
print("CollecTRI columns:", df_tf.columns.tolist())
tf_src_col = df_tf.columns[0]  # typically 'source'
tf_tgt_col = df_tf.columns[1]  # typically 'target'
df_tf["tf_sym"] = df_tf[tf_src_col].map(norm_sym)
df_tf["tgt_sym"] = df_tf[tf_tgt_col].map(norm_sym)
df_tf = df_tf[(df_tf["tf_sym"] != "") & (df_tf["tgt_sym"] != "")].copy()

# ==========================================
# 3. Collect & Partition Node Vocabularies
# ==========================================
lnc_syms, mrna_syms, mirna_syms, prot_syms = set(), set(), set(), set()

for _, r in df_rr[["sym1", "cat1", "sym2", "cat2"]].iterrows():
    if r["cat1"] == "lncrna": lnc_syms.add(r["sym1"])
    elif r["cat1"] == "mrna": mrna_syms.add(r["sym1"])
    elif r["cat1"] == "mirna": mirna_syms.add(r["sym1"])
    if r["cat2"] == "lncrna": lnc_syms.add(r["sym2"])
    elif r["cat2"] == "mrna": mrna_syms.add(r["sym2"])
    elif r["cat2"] == "mirna": mirna_syms.add(r["sym2"])

for _, r in df_rp[["sym1", "cat1", "sym2"]].iterrows():
    if r["cat1"] == "lncrna": lnc_syms.add(r["sym1"])
    elif r["cat1"] == "mrna": mrna_syms.add(r["sym1"])
    elif r["cat1"] == "mirna": mirna_syms.add(r["sym1"])
    prot_syms.add(r["sym2"])

# Include CollecTRI TF -> target nodes (TFs as proteins, targets as mRNAs/lncRNAs)
prot_syms.update(df_tf["tf_sym"])
for tgt in df_tf["tgt_sym"]:
    if tgt not in lnc_syms and tgt in expr_symbols:
        mrna_syms.add(tgt)

# Disambiguate RNA categories (lncrna > mirna > mrna)
mirna_syms -= lnc_syms
mrna_syms -= (lnc_syms | mirna_syms)

# Anchor lncRNA and mRNA to expression-backed genes
lnc_list = sorted(lnc_syms & expr_symbols)
mrna_list = sorted(mrna_syms & expr_symbols)
mirna_list = sorted(mirna_syms)
prot_list = sorted(prot_syms)

# Diseases from diseasome + LncRNADisease3
dis_labels = pd.read_csv(os.path.join(COMB_DIR, "disease.labels.txt"), sep=",")
print("disease.labels.txt columns:", dis_labels.columns.tolist())

node_maps = {
    "lncrna": {s: i for i, s in enumerate(lnc_list)},
    "mrna": {s: i for i, s in enumerate(mrna_list)},
    "mirna": {s: i for i, s in enumerate(mirna_list)},
    "protein": {s: i for i, s in enumerate(prot_list)},
}

# ==========================================
# 4. Build Node Feature Tensors
# ==========================================
def compute_cohort_features(sym_list):
    sub = df_gene_expr.loc[sym_list]
    g = sub[gtex_cols].values
    l = sub[lgg_cols].values
    b = sub[gbm_cols].values

    feats = np.column_stack([
        g.mean(axis=1), g.std(axis=1), np.median(g, axis=1),
        l.mean(axis=1), l.std(axis=1), np.median(l, axis=1),
        b.mean(axis=1), b.std(axis=1), np.median(b, axis=1),
        l.mean(axis=1) - g.mean(axis=1),  # LGG vs Normal Brain log2FC
        b.mean(axis=1) - g.mean(axis=1),  # GBM vs Normal Brain log2FC
        b.mean(axis=1) - l.mean(axis=1),  # GBM vs LGG log2FC
    ])
    feats = (feats - feats.mean(axis=0, keepdims=True)) / (feats.std(axis=0, keepdims=True) + 1e-6)
    return torch.tensor(feats, dtype=torch.float32)

data = HeteroData()
data["lncrna"].x = compute_cohort_features(lnc_list)
data["mrna"].x = compute_cohort_features(mrna_list)

# Protein features: map matched gene expression where available, zero-impute + mask otherwise
prot_expr_syms = [s for s in prot_list if s in expr_symbols]
prot_expr_map = dict(zip(prot_expr_syms, compute_cohort_features(prot_expr_syms).numpy()))
prot_feats = np.zeros((len(prot_list), 13), dtype=np.float32)
for i, s in enumerate(prot_list):
    if s in prot_expr_map:
        prot_feats[i, :12] = prot_expr_map[s]
        prot_feats[i, 12] = 1.0  # binary indicator for observed mRNA expression
data["protein"].x = torch.tensor(prot_feats, dtype=torch.float32)

data["mirna"].num_nodes = len(mirna_list)
data["mirna"].node_id = torch.arange(len(mirna_list), dtype=torch.long)

# ==========================================
# 5. Assemble Edges (RNA-RNA, RNA-Protein, TF-Gene)
# ==========================================
edge_buckets = {}

# 5a. Undirected RNA-RNA interactions
for _, r in df_rr.iterrows():
    s1, c1, s2, c2 = r["sym1"], r["cat1"], r["sym2"], r["cat2"]
    if s1 in node_maps[c1] and s2 in node_maps[c2]:
        u, v = node_maps[c1][s1], node_maps[c2][s2]
        if (c1, u) > (c2, v):
            c1, u, c2, v = c2, v, c1, u
        edge_buckets.setdefault((c1, "interacts", c2), set()).add((u, v))

# 5b. RNA-Protein binding (Download_data_RP.txt)
for _, r in df_rp.iterrows():
    s1, c1, s2 = r["sym1"], r["cat1"], r["sym2"]
    if s1 in node_maps[c1] and s2 in node_maps["protein"]:
        u, v = node_maps[c1][s1], node_maps["protein"][s2]
        edge_buckets.setdefault((c1, "binds", "protein"), set()).add((u, v))

# 5c. Directed TF -> Target Gene regulation (CollecTRI)
for _, r in df_tf.iterrows():
    tf, tgt = r["tf_sym"], r["tgt_sym"]
    if tf in node_maps["protein"]:
        u = node_maps["protein"][tf]
        if tgt in node_maps["mrna"]:
            edge_buckets.setdefault(("protein", "regulates", "mrna"), set()).add((u, node_maps["mrna"][tgt]))
        elif tgt in node_maps["lncrna"]:
            edge_buckets.setdefault(("protein", "regulates", "lncrna"), set()).add((u, node_maps["lncrna"][tgt]))

for rel, pairs in edge_buckets.items():
    edges = sorted(pairs)
    data[rel].edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()

# Add reverse edges for message passing (keeps directed 'regulates' distinct from 'rev_regulates')
data = T.ToUndirected()(data)

out_path = os.path.join(OUT_DIR, "multiomic_causal_hetero.pt")
torch.save({"graph": data, "node_maps": node_maps}, out_path)
print("\n=== Saved HeteroData ===")
print(data)
