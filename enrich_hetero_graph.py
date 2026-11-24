import os
import re
import urllib.request
import torch
import numpy as np
import pandas as pd
import torch_geometric.transforms as T

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
XENA_DIR = os.path.join(BASE_DIR, "data/xena_toil")
COMB_DIR = os.path.join(BASE_DIR, "data/lncRNA_combined")
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUT_DIR = os.path.join(BASE_DIR, "data/processed")

def norm_sym(s):
    if pd.isna(s):
        return ""
    s = str(s).strip()
    s = re.sub(r"^has-mir-", "hsa-miR-", s, flags=re.IGNORECASE)
    s = re.sub(r"^hsa-mir-", "hsa-miR-", s, flags=re.IGNORECASE)
    return s.upper()

# 1. Load existing base graph & node maps
bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]

lnc_list = sorted(node_maps["lncrna"], key=node_maps["lncrna"].get)
mrna_list = sorted(node_maps["mrna"], key=node_maps["mrna"].get)
prot_list = sorted(node_maps["protein"], key=node_maps["protein"].get)

# 2. Enrich lncRNA, mRNA, and Protein with 99-patient CPTAC-GBM RNA & TF Proteomics
gbm_rna = pd.read_parquet(os.path.join(RAW_DIR, "gbm_rna.parquet"))
gbm_rna.columns = [norm_sym(c) for c in gbm_rna.columns]
gbm_rna = gbm_rna.groupby(level=0, axis=1).mean()

gbm_prot = pd.read_parquet(os.path.join(RAW_DIR, "gbm_tf_prot.parquet"))
gbm_prot.columns = [norm_sym(c) for c in gbm_prot.columns]
gbm_prot = gbm_prot.groupby(level=0, axis=1).mean()

def get_gbm_stats(df_mat, sym_list):
    n = len(sym_list)
    feats = np.zeros((n, 5), dtype=np.float32)  # mean, std, median, iqr, observed_mask
    col_set = set(df_mat.columns)
    matched = [s for s in sym_list if s in col_set]
    if matched:
        vals = df_mat[matched].values  # (99, len(matched))
        means = np.nanmean(vals, axis=0)
        stds = np.nanstd(vals, axis=0)
        meds = np.nanmedian(vals, axis=0)
        iqrs = np.nanpercentile(vals, 75, axis=0) - np.nanpercentile(vals, 25, axis=0)
        # Z-score across matched genes
        stats = np.column_stack([means, stds, meds, iqrs])
        stats = (stats - stats.mean(axis=0, keepdims=True)) / (stats.std(axis=0, keepdims=True) + 1e-6)
        for idx_m, s in enumerate(matched):
            row_i = sym_list.index(s)
            feats[row_i, :4] = stats[idx_m]
            feats[row_i, 4] = 1.0
    return torch.tensor(feats, dtype=torch.float32)

data["lncrna"].x = torch.cat([data["lncrna"].x, get_gbm_stats(gbm_rna, lnc_list)], dim=1)
data["mrna"].x = torch.cat([data["mrna"].x, get_gbm_stats(gbm_rna, mrna_list)], dim=1)
data["protein"].x = torch.cat([data["protein"].x, get_gbm_stats(gbm_prot, prot_list)], dim=1)

# 3. Attach Signed CollecTRI Edge Weights (+1 / -1)
df_tf = pd.read_csv(os.path.join(RAW_DIR, "collectri_human.csv"))
df_tf["tf_sym"] = df_tf["source"].map(norm_sym)
df_tf["tgt_sym"] = df_tf["target"].map(norm_sym)
tf_sign_map = dict(zip(zip(df_tf["tf_sym"], df_tf["tgt_sym"]), df_tf["weight"].astype(float)))

for tgt_type in ["mrna", "lncrna"]:
    rel = ("protein", "regulates", tgt_type)
    rev_rel = (tgt_type, "rev_regulates", "protein")
    ei = data[rel].edge_index
    tgt_list = mrna_list if tgt_type == "mrna" else lnc_list
    weights = [
        tf_sign_map.get((prot_list[u.item()], tgt_list[v.item()]), 1.0)
        for u, v in zip(ei[0], ei[1])
    ]
    w_tensor = torch.tensor(weights, dtype=torch.float32).unsqueeze(1)
    data[rel].edge_attr = w_tensor
    data[rev_rel].edge_attr = w_tensor.clone()

# 4. Add Disease Nodes (Diseasome + LncRNADisease v3.0 vocabulary)
dis_df = pd.read_csv(os.path.join(COMB_DIR, "disease.labels.txt"), sep=",")
dis_edges_df = pd.read_csv(os.path.join(COMB_DIR, "diseasome.edgelist.txt"), sep=" ")
lnc_dis_df = pd.read_csv(os.path.join(COMB_DIR, "all_ncRNA_disease_information_LncRNADisease3.tsv"), sep="\t", low_memory=False)

def norm_dis(s):
    return re.sub(r"\s+", " ", str(s).strip().lower()) if pd.notna(s) else ""

dis_names = set(dis_df["Label"].map(norm_dis)) | set(lnc_dis_df["Disease Name"].map(norm_dis))
dis_names.discard("")
dis_list = sorted(dis_names)
node_maps["disease"] = {d: i for i, d in enumerate(dis_list)}

data["disease"].num_nodes = len(dis_list)
data["disease"].node_id = torch.arange(len(dis_list), dtype=torch.long)

id_to_dis = dict(zip(dis_df["Id"], dis_df["Label"].map(norm_dis)))
dis_edge_set = set()
for _, r in dis_edges_df.iterrows():
    u_name = id_to_dis.get(r.iloc[0], "")
    v_name = id_to_dis.get(r.iloc[1], "")
    if u_name in node_maps["disease"] and v_name in node_maps["disease"] and u_name != v_name:
        u, v = node_maps["disease"][u_name], node_maps["disease"][v_name]
        dis_edge_set.add((u, v))
        dis_edge_set.add((v, u))

if dis_edge_set:
    data["disease", "comorbid_with", "disease"].edge_index = (
        torch.tensor(sorted(dis_edge_set), dtype=torch.long).t().contiguous()
    )

# 5. Add Drug Nodes (ChG-Miner via HGNC Entrez -> Symbol mapping)
hgnc_path = os.path.join(RAW_DIR, "hgnc_complete_set.txt")
if not os.path.exists(hgnc_path):
    urllib.request.urlretrieve(
        "https://storage.googleapis.com/public-download-files/hgnc/tsv/tsv/hgnc_complete_set.txt",
        hgnc_path
    )
hgnc = pd.read_csv(hgnc_path, sep="\t", usecols=["symbol", "entrez_id"], low_memory=False).dropna()
entrez_to_sym = dict(zip(hgnc["entrez_id"].astype(int).astype(str), hgnc["symbol"].map(norm_sym)))

chg = pd.read_csv(os.path.join(COMB_DIR, "ChG-Miner_miner-chem-gene.tsv"), sep="\t")
chg.columns = ["drug", "entrez_id"]
chg["gene_sym"] = chg["entrez_id"].astype(str).str.strip().map(entrez_to_sym)
chg = chg.dropna(subset=["drug", "gene_sym"])
chg = chg[chg["gene_sym"].isin(node_maps["protein"])].copy()

drug_list = sorted(chg["drug"].unique())
node_maps["drug"] = {d: i for i, d in enumerate(drug_list)}
data["drug"].num_nodes = len(drug_list)
data["drug"].node_id = torch.arange(len(drug_list), dtype=torch.long)

drug_prot_edges = sorted({
    (node_maps["drug"][r["drug"]], node_maps["protein"][r["gene_sym"]])
    for _, r in chg.iterrows()
})
dp_ei = torch.tensor(drug_prot_edges, dtype=torch.long).t().contiguous()
data["drug", "targets", "protein"].edge_index = dp_ei
data["protein", "rev_targets", "drug"].edge_index = dp_ei.flip(0)

out_path = os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt")
torch.save({"graph": data, "node_maps": node_maps}, out_path)
print("=== Full Multi-Omic HeteroData ===")
print(data)
