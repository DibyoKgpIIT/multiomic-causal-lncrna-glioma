import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, Linear
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
OUT_DIR = os.path.join(BASE_DIR, "data/processed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]
splits = torch.load(os.path.join(OUT_DIR, "multitask_splits.pt"), weights_only=False)

# Remove empty edge types and strip edge_attr on CollecTRI (unsigned SAGEConv across all relations)
for et in list(data.edge_types):
    if data[et].edge_index.numel() == 0:
        del data[et]
    elif "edge_attr" in data[et]:
        del data[et].edge_attr

idx_to_lnc = {v: k for k, v in node_maps["lncrna"].items()}
n_lnc = data["lncrna"].num_nodes
valid_lnc_mask = data["lncrna"].strict_lnc_mask

# Leak-free entity-disjoint splits
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

data = data.to(device)
y_drv_d = y_drv.to(device)
y_dir_d = y_dir.to(device)
inner_tr_d = inner_tr_idx.to(device)
inner_va_d = inner_va_idx.to(device)
val_d = val_idx.to(device)
test_d = test_idx.to(device)


class UnsignedMultiOmicCausalGNN(nn.Module):
    def __init__(self, metadata, hidden_dim=64, dropout=0.35):
        super().__init__()
        self.dropout = dropout
        self.feat_proj = nn.ModuleDict({
            "lncrna": Linear(data["lncrna"].x.size(1), hidden_dim),
            "mrna": Linear(data["mrna"].x.size(1), hidden_dim),
            "protein": Linear(data["protein"].x.size(1), hidden_dim),
        })
        self.id_emb = nn.ModuleDict({
            nt: nn.Embedding(data[nt].num_nodes, hidden_dim)
            for nt in ["mirna", "disease", "drug"]
            if nt in data.node_types and data[nt].num_nodes > 0
        })

        # Pure unsigned SAGEConv message passing across all relations
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(2):
            conv_dict = {et: SAGEConv((-1, -1), hidden_dim) for et in metadata[1]}
            self.convs.append(HeteroConv(conv_dict, aggr="mean"))
            self.norms.append(nn.ModuleDict({nt: nn.LayerNorm(hidden_dim) for nt in metadata[0]}))

        self.link_head = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )
        self.driver_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )
        self.polarity_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),
        )

    def encode_nodes(self):
        h = {}
        for nt in data.node_types:
            if nt in self.feat_proj:
                h[nt] = F.gelu(self.feat_proj[nt](data[nt].x))
            elif nt in self.id_emb:
                h[nt] = self.id_emb[nt](data[nt].node_id)

        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, data.edge_index_dict)
            for nt in h:
                if nt in h_next:
                    h[nt] = norm[nt](h[nt] + F.dropout(F.gelu(h_next[nt]), p=self.dropout, training=self.training))
        return h

    def decode_links(self, h_lnc, h_dis, edge_index):
        src = h_lnc[edge_index[0]]
        dst = h_dis[edge_index[1]]
        return self.link_head(torch.cat([src, dst, src * dst], dim=-1)).squeeze(-1)


torch.manual_seed(42)
model = UnsignedMultiOmicCausalGNN(data.metadata(), hidden_dim=64, dropout=0.35).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-3)

pos_weight_drv = torch.tensor([(y_drv_d[inner_tr_d] == 0).sum() / (y_drv_d[inner_tr_d] == 1).sum()], device=device)
pol_tr = inner_tr_d[y_dir_d[inner_tr_d] >= 0]
pos_weight_pol = torch.tensor([(y_dir_d[pol_tr] == 0).sum() / (y_dir_d[pol_tr] == 1).sum()], device=device)


@torch.no_grad()
def evaluate(ei_t1, y_t1, w_t1, idx_t2):
    model.eval()
    h = model.encode_nodes()
    p1 = torch.sigmoid(model.decode_links(h["lncrna"], h["disease"], ei_t1)).cpu().numpy()
    y1, w1 = y_t1.cpu().numpy(), w_t1.cpu().numpy()
    auc_all = roc_auc_score(y1, p1)
    auc_hard = roc_auc_score(y1[w1 == 1.0], p1[w1 == 1.0])

    p2 = torch.sigmoid(model.driver_head(h["lncrna"][idx_t2]).squeeze(-1)).cpu().numpy()
    y2 = y_drv_d[idx_t2].cpu().numpy()
    auc_drv = roc_auc_score(y2, p2)
    ap_drv = average_precision_score(y2, p2)
    return auc_all, auc_hard, auc_drv, ap_drv


best_hmean = -1.0
best_ep = 0
ckpt_path = os.path.join(OUT_DIR, "multitask_causal_gnn_pareto.pt")

for epoch in range(1, 41):
    model.train()
    optimizer.zero_grad()
    h = model.encode_nodes()

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
    optimizer.step()

    _, va_hard, va_drv, _ = evaluate(t1_va_ei, t1_va_y, t1_va_w, inner_va_d)
    hmean = 2 * va_hard * va_drv / (va_hard + va_drv + 1e-8)
    if hmean > best_hmean:
        best_hmean = hmean
        best_ep = epoch
        torch.save(model.state_dict(), ckpt_path)

print(f"Selected Pareto-Optimal Checkpoint at Epoch {best_ep} (Inner-Val H-Mean: {best_hmean:.4f})")
model.load_state_dict(torch.load(ckpt_path, weights_only=True))

va_all, va_hard, va_drv, va_ap = evaluate(t1_va_ei, t1_va_y, t1_va_w, val_d)
te_all, te_hard, te_drv, te_ap = evaluate(t1_te_ei, t1_te_y, t1_te_w, test_d)
print(f"Held-Out VAL  -> Link AUROC: {va_all:.4f} (Causal-vs-No: {va_hard:.4f}) | Glioma Driver AUROC: {va_drv:.4f} / AUPRC: {va_ap:.4f}")
print(f"Held-Out TEST -> Link AUROC: {te_all:.4f} (Causal-vs-No: {te_hard:.4f}) | Glioma Driver AUROC: {te_drv:.4f} / AUPRC: {te_ap:.4f}")

# Generate clean strict non-coding candidate rankings
true_glioma_terms = {"brain glioma", "diffuse intrinsic pontine glioma", "glioblastoma", "glioma", "oligodendroglioma"}
glioma_dis_ids = [idx for name, idx in node_maps["disease"].items() if name in true_glioma_terms]

model.eval()
with torch.no_grad():
    h = model.encode_nodes()
    p_driver = torch.sigmoid(model.driver_head(h["lncrna"]).squeeze(-1)).cpu()
    p_polarity = torch.sigmoid(model.polarity_head(h["lncrna"]).squeeze(-1)).cpu()
    p_links = [
        torch.sigmoid(
            model.decode_links(
                h["lncrna"],
                h["disease"],
                torch.stack([torch.arange(n_lnc, device=device), torch.full((n_lnc,), d_id, device=device)]),
            )
        ).cpu()
        for d_id in glioma_dis_ids
    ]
    p_glioma_link = torch.stack(p_links, dim=0).max(dim=0).values

rows = []
for i in range(n_lnc):
    if not valid_lnc_mask[i]:
        continue
    status = "Known_Glioma_Driver" if y_drv[i] == 1 else ("NonGlioma_Causal" if y_drv[i] == 0 else "Unannotated")
    rows.append({
        "symbol": idx_to_lnc[i],
        "status": status,
        "joint_causal_score": float(0.5 * p_driver[i] + 0.5 * p_glioma_link[i]),
        "driver_prob": float(p_driver[i]),
        "glioma_link_prob": float(p_glioma_link[i]),
        "oncogenic_prob": float(p_polarity[i]),
        "predicted_role": "Oncogenic (Up)" if p_polarity[i] >= 0.5 else "Suppressive (Down)",
    })

df_out = pd.DataFrame(rows).sort_values("joint_causal_score", ascending=False)
df_out.to_csv(os.path.join(OUT_DIR, "glioma_causal_predictions_clean.csv"), index=False)

print("\n=== TOP 15 NOVEL BONAFIDE LNCRNA CANDIDATES (UNSIGNED COLLECTRI, NO SKIP) ===")
print(df_out[df_out["status"] == "Unannotated"].head(15)[
    ["symbol", "joint_causal_score", "driver_prob", "glioma_link_prob", "oncogenic_prob", "predicted_role"]
].to_string(index=False))
