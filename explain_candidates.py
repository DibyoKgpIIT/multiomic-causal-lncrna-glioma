
import os
import torch
import pandas as pd

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
OUT_DIR = os.path.join(BASE_DIR, "data/processed")

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
#bundle = torch.load(os.path.join(OUT_DIR, "multitask_causal_gnn_pareto.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]

idx_to_lnc = {v: k for k, v in node_maps["lncrna"].items()}
idx_to_prot = {v: k for k, v in node_maps["protein"].items()}
idx_to_mir = {v: k for k, v in node_maps["mirna"].items()}
idx_to_mrna = {v: k for k, v in node_maps["mrna"].items()}
idx_to_drug = {v: k for k, v in node_maps["drug"].items()}

df_pred = pd.read_csv(os.path.join(OUT_DIR, "glioma_causal_predictions_clean.csv"))

# Apply strict HGNC non-coding mask (removes HLA-DRB1, MS4A6A, MT1X, pseudogenes)
strict_mask = data["lncrna"].strict_lnc_mask.numpy()
strict_syms = {idx_to_lnc[i] for i, keep in enumerate(strict_mask) if keep}

df_strict = df_pred[df_pred["symbol"].isin(strict_syms)].copy()
df_strict.to_csv(os.path.join(OUT_DIR, "glioma_causal_predictions_strict_lncrna.csv"), index=False)

novel_top10 = df_strict[df_strict["status"] == "Unannotated"].head(10)
print("=== TOP 10 STRICTLY VERIFIED NON-CODING LNCRNA CANDIDATES ===")
print(novel_top10[["symbol", "joint_causal_score", "driver_prob", "glioma_link_prob", "oncogenic_prob", "predicted_role"]].to_string(index=False))

# Extract mechanistic subgraphs (TF -> lncRNA, lncRNA <-> miRNA, lncRNA <-> mRNA, Drug -> TF)
tf_ei = data["protein", "regulates", "lncrna"].edge_index.numpy()
tf_w = data["protein", "regulates", "lncrna"].edge_attr.squeeze(-1).numpy()
mir_ei = data["lncrna", "interacts", "mirna"].edge_index.numpy()
mrna_ei = data["lncrna", "interacts", "mrna"].edge_index.numpy()
dp_ei = data["drug", "targets", "protein"].edge_index.numpy()

prot_to_drugs = {}
for d_i, p_i in zip(dp_ei[0], dp_ei[1]):
    prot_to_drugs.setdefault(p_i, []).append(idx_to_drug[d_i])

print("\n=== MECHANISTIC CAUSAL PATHWAYS FOR TOP 10 NOVEL LNCRNAS ===")
for sym in novel_top10["symbol"]:
    u = node_maps["lncrna"][sym]
    tfs = [(idx_to_prot[p], w, p) for p, l, w in zip(tf_ei[0], tf_ei[1], tf_w) if l == u]
    mirs = [idx_to_mir[m] for l, m in zip(mir_ei[0], mir_ei[1]) if l == u][:5]
    mrnas = [idx_to_mrna[m] for l, m in zip(mrna_ei[0], mrna_ei[1]) if l == u][:5]
    drugs = sorted({d for _, _, p in tfs for d in prot_to_drugs.get(p, [])})[:5]
    tf_str = ", ".join([f"{t}({'+' if w > 0 else '-'})" for t, w, _ in tfs[:5]]) or "None"
    print(f"\n[{sym}]")
    print(f"  Upstream CollecTRI TFs : {tf_str}")
    print(f"  Interacting miRNAs     : {', '.join(mirs) if mirs else 'None'}")
    print(f"  Target/Co-expr mRNAs   : {', '.join(mrnas) if mrnas else 'None'}")
    print(f"  Upstream Drugs (via TF): {', '.join(drugs) if drugs else 'None'}")
