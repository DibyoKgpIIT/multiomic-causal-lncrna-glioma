import os
import re
import torch
import pandas as pd

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
COMB_DIR = os.path.join(BASE_DIR, "data/lncRNA_combined")
RAW_DIR = os.path.join(BASE_DIR, "data/raw")
OUT_DIR = os.path.join(BASE_DIR, "data/processed")

def norm_sym(s):
    return str(s).strip().upper() if pd.notna(s) else ""

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]

chg = pd.read_csv(os.path.join(COMB_DIR, "ChG-Miner_miner-chem-gene.tsv"), sep="\t")
chg.columns = ["drug", "gene_id"]
print("Sample ChG-Miner rows:\n", chg.head(5))

hgnc = pd.read_csv(
    os.path.join(RAW_DIR, "hgnc_complete_set.txt"),
    sep="\t",
    usecols=["symbol", "entrez_id", "uniprot_ids"],
    low_memory=False
)

# Build multi-key lookup: symbol, entrez_id, and uniprot_ids -> HGNC symbol
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

drug_prot_edges = sorted({
    (node_maps["drug"][r["drug"]], node_maps["protein"][r["gene_sym"]])
    for _, r in chg.iterrows()
})
dp_ei = torch.tensor(drug_prot_edges, dtype=torch.long).t().contiguous()
data["drug", "targets", "protein"].edge_index = dp_ei
data["protein", "rev_targets", "drug"].edge_index = dp_ei.flip(0)

torch.save({"graph": data, "node_maps": node_maps}, os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"))
print(f"Populated {len(drug_list)} drugs and {dp_ei.size(1)} drug-protein edges.")
