import os
import torch
import pandas as pd

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
OUT_DIR = os.path.join(BASE_DIR, "data/processed")

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]

idx_to_lnc = {v: k for k, v in node_maps["lncrna"].items()}
idx_to_prot = {v: k for k, v in node_maps["protein"].items()}
idx_to_mir = {v: k for k, v in node_maps["mirna"].items()}
idx_to_mrna = {v: k for k, v in node_maps["mrna"].items()}
idx_to_drug = {v: k for k, v in node_maps["drug"].items()}

df_strict = pd.read_csv(os.path.join(OUT_DIR, "glioma_causal_predictions_strict_lncrna.csv"))
novel_top10 = df_strict[df_strict["status"] == "Unannotated"].head(10)

# Build adjacency lookups
bind_ei = data["lncrna", "binds", "protein"].edge_index.numpy()
lnc_to_prots = {}
for l_i, p_i in zip(bind_ei[0], bind_ei[1]):
    lnc_to_prots.setdefault(l_i, []).append(p_i)

tf_mrna_ei = data["protein", "regulates", "mrna"].edge_index.numpy()
tf_mrna_w = data["protein", "regulates", "mrna"].edge_attr.squeeze(-1).numpy()
tf_set = set(tf_mrna_ei[0].tolist())  # proteins that are active CollecTRI TFs
mrna_to_tfs = {}
for p_i, m_i, w in zip(tf_mrna_ei[0], tf_mrna_ei[1], tf_mrna_w):
    mrna_to_tfs.setdefault(m_i, []).append((p_i, w))

mir_ei = data["lncrna", "interacts", "mirna"].edge_index.numpy()
lnc_to_mirs = {}
for l_i, m_i in zip(mir_ei[0], mir_ei[1]):
    lnc_to_mirs.setdefault(l_i, []).append(m_i)

mrna_ei = data["lncrna", "interacts", "mrna"].edge_index.numpy()
lnc_to_mrnas = {}
for l_i, m_i in zip(mrna_ei[0], mrna_ei[1]):
    lnc_to_mrnas.setdefault(l_i, []).append(m_i)

dp_ei = data["drug", "targets", "protein"].edge_index.numpy()
prot_to_drugs = {}
for d_i, p_i in zip(dp_ei[0], dp_ei[1]):
    prot_to_drugs.setdefault(p_i, []).append(idx_to_drug[d_i])

print("=== 1-HOP & 2-HOP MECHANISTIC CAUSAL SUBGRAPHS FOR TOP 10 NOVEL LNCRNAS ===")
for sym in novel_top10["symbol"]:
    u = node_maps["lncrna"][sym]
    bound_prots = lnc_to_prots.get(u, [])
    bound_tfs = [p for p in bound_prots if p in tf_set]
    bound_rbps = [p for p in bound_prots if p not in tf_set]

    mirs = [idx_to_mir[m] for m in lnc_to_mirs.get(u, [])[:5]]
    mrnas = lnc_to_mrnas.get(u, [])

    # 2-hop TFs regulating the co-interacting mRNAs
    two_hop_tfs = sorted({
        (idx_to_prot[p], w, p)
        for m in mrnas
        for p, w in mrna_to_tfs.get(m, [])
    }, key=lambda x: x[0])

    # Drugs targeting either directly bound TFs/proteins or 2-hop TFs
    candidate_prots = bound_tfs[:5] + [p for _, _, p in two_hop_tfs[:5]]
    drugs = sorted({d for p in candidate_prots for d in prot_to_drugs.get(p, [])})[:6]

    print(f"\n[{sym}]")
    print(f"  1-Hop Bound TFs (RNAInter) : {', '.join([idx_to_prot[p] for p in bound_tfs[:6]]) or 'None'}")
    print(f"  1-Hop Bound RBPs/Proteins  : {', '.join([idx_to_prot[p] for p in bound_rbps[:6]]) or 'None'}")
    print(f"  1-Hop Sponge miRNAs        : {', '.join(mirs) or 'None'}")
    print(f"  1-Hop Partner mRNAs        : {', '.join([idx_to_mrna[m] for m in mrnas[:5]]) or 'None'}")
    #print(f"  2-Hop CollecTRI TFs        : {', '.join([f"{t}({'+' if w>0 else '-'})" for t, w, _ in two_hop_tfs[:6]]) or 'None'}")
    tf_2hop_str = ", ".join([f"{t}({'+' if w > 0 else '-'})" for t, w, _ in two_hop_tfs[:6]]) or "None"
    print(f"  2-Hop CollecTRI TFs        : {tf_2hop_str}")
    print(f"  Linked DrugBank Compounds  : {', '.join(drugs) or 'None'}")
