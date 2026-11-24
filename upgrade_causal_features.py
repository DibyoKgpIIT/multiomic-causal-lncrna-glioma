import os
import re
import torch
import numpy as np
import pandas as pd

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUT_DIR = os.path.join(BASE_DIR, "data/processed")

def norm_sym(s):
    return str(s).strip().upper() if pd.notna(s) else ""

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]

lnc_list = sorted(node_maps["lncrna"], key=node_maps["lncrna"].get)
mrna_list = sorted(node_maps["mrna"], key=node_maps["mrna"].get)
n_lnc = len(lnc_list)

# 1. Strict HGNC non-coding RNA biotype mask (excludes protein-coding HLA/MS4A6A/SEC61G)
hgnc = pd.read_csv(
    os.path.join(RAW_DIR, "hgnc_complete_set.txt"),
    sep="\t",
    usecols=["symbol", "locus_group", "locus_type"],
    low_memory=False
).dropna(subset=["symbol"])
hgnc["sym"] = hgnc["symbol"].map(norm_sym)
coding_syms = set(hgnc.loc[hgnc["locus_group"] == "protein-coding gene", "sym"])
pseudo_syms = set(hgnc.loc[hgnc["locus_group"] == "pseudogene", "sym"])

strict_lnc_mask = torch.tensor(
    [(s not in coding_syms) and (s not in pseudo_syms) for s in lnc_list],
    dtype=torch.bool
)
print(f"Strict HGNC non-coding mask: {int(strict_lnc_mask.sum())} / {n_lnc} (removed coding & pseudogenes)")

# 2. Compute 99-patient CPTAC-GBM TF-Protein <-> lncRNA empirical causal coupling
gbm_rna = pd.read_parquet(os.path.join(RAW_DIR, "gbm_rna.parquet"))
gbm_rna.columns = [norm_sym(c) for c in gbm_rna.columns]
gbm_rna = gbm_rna.T.groupby(level=0).mean().T  # (99, genes)

gbm_prot = pd.read_parquet(os.path.join(RAW_DIR, "gbm_tf_prot.parquet"))
gbm_prot.columns = [norm_sym(c) for c in gbm_prot.columns]
gbm_prot = gbm_prot.T.groupby(level=0).mean().T  # (99, 684 TFs)

# Standardize patient columns for fast vectorized Pearson correlation
tf_mat = gbm_prot.values.astype(np.float32)
tf_mat = (tf_mat - np.nanmean(tf_mat, axis=0, keepdims=True)) / (np.nanstd(tf_mat, axis=0, keepdims=True) + 1e-6)
tf_mat = np.nan_to_num(tf_mat, nan=0.0)
tf_cols = list(gbm_prot.columns)
tf_col_idx = {s: i for i, s in enumerate(tf_cols)}

# Load CollecTRI signed TF -> lncRNA edges
df_tf = pd.read_csv(os.path.join(RAW_DIR, "collectri_human.csv"))
df_tf["tf_sym"] = df_tf["source"].map(norm_sym)
df_tf["tgt_sym"] = df_tf["target"].map(norm_sym)
collectri_by_lnc = {}
for _, r in df_tf.iterrows():
    if r["tf_sym"] in tf_col_idx:
        collectri_by_lnc.setdefault(r["tgt_sym"], []).append((tf_col_idx[r["tf_sym"]], float(r["weight"])))

rna_col_set = set(gbm_rna.columns)
causal_feats = np.zeros((n_lnc, 4), dtype=np.float32)  # [max_abs_tf_r, top5_abs_tf_r, collectri_signed_r, in_degree_tf]

for i, sym in enumerate(lnc_list):
    reg_tfs = collectri_by_lnc.get(sym, [])
    causal_feats[i, 3] = np.log1p(len(reg_tfs))
    if sym in rna_col_set:
        y = gbm_rna[sym].values.astype(np.float32)
        y = (y - np.nanmean(y)) / (np.nanstd(y) + 1e-6)
        y = np.nan_to_num(y, nan=0.0)
        corrs = (tf_mat.T @ y) / len(y)  # (684,) Pearson r with all 684 TF proteins
        abs_c = np.abs(corrs)
        causal_feats[i, 0] = float(np.max(abs_c))
        causal_feats[i, 1] = float(np.mean(np.sort(abs_c)[-5:]))
        if reg_tfs:
            # Agreement between empirical patient TF-protein correlation and CollecTRI prior sign
            signed_scores = [corrs[t_idx] * w for t_idx, w in reg_tfs]
            causal_feats[i, 2] = float(np.mean(signed_scores))

causal_feats = (causal_feats - causal_feats.mean(axis=0, keepdims=True)) / (causal_feats.std(axis=0, keepdims=True) + 1e-6)

# Save updated feature tensor (17 + 4 = 21 dims) and strict mask
if data["lncrna"].x.size(1) == 17:
    data["lncrna"].x = torch.cat([data["lncrna"].x, torch.tensor(causal_feats, dtype=torch.float32)], dim=1)
data["lncrna"].strict_lnc_mask = strict_lnc_mask

torch.save({"graph": data, "node_maps": node_maps}, os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"))
print("Updated lncRNA feature shape:", data["lncrna"].x.shape)
