import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, Linear
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
OUT_DIR = os.path.join(BASE_DIR, "data/processed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
data = bundle["graph"]
splits = torch.load(os.path.join(OUT_DIR, "multitask_splits.pt"), weights_only=False)

for et in list(data.edge_types):
    if data[et].edge_index.numel() == 0:
        del data[et]

data = data.to(device)

class MultiOmicCausalGNN(nn.Module):
    def __init__(self, metadata, hidden_dim=64, dropout=0.25):
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

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(2):
            conv_dict = {}
            for et in metadata[1]:
                if "regulates" in et[1] and "edge_attr" in data[et]:
                    conv_dict[et] = GATv2Conv((-1, -1), hidden_dim, edge_dim=1, add_self_loops=False)
                else:
                    conv_dict[et] = SAGEConv((-1, -1), hidden_dim)
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

        edge_attr_dict = {
            et: data[et].edge_attr
            for et in data.edge_types
            if "edge_attr" in data[et]
        }

        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, data.edge_index_dict, edge_attr_dict=edge_attr_dict)
            for nt in h:
                if nt in h_next:
                    h[nt] = norm[nt](h[nt] + F.dropout(F.gelu(h_next[nt]), p=self.dropout, training=self.training))
        return h

    def decode_links(self, h_lnc, h_dis, edge_index):
        src = h_lnc[edge_index[0]]
        dst = h_dis[edge_index[1]]
        return self.link_head(torch.cat([src, dst, src * dst], dim=-1)).squeeze(-1)

model = MultiOmicCausalGNN(data.metadata(), hidden_dim=64).to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-4)

# Move split tensors to device
t1_tr_ei, t1_tr_y, t1_tr_w = [x.to(device) for x in splits["task1"]["train"]]
t1_va_ei, t1_va_y, t1_va_w = [x.to(device) for x in splits["task1"]["val"]]
t1_te_ei, t1_te_y, t1_te_w = [x.to(device) for x in splits["task1"]["test"]]

y_drv = splits["task2"]["y_driver"].to(device)
y_dir = splits["task2"]["y_dir"].to(device)
tr_idx = splits["task2"]["train_idx"].to(device)
va_idx = splits["task2"]["val_idx"].to(device)
te_idx = splits["task2"]["test_idx"].to(device)

# Upweight minority positive glioma drivers (~255 pos vs 527 neg)
pos_weight_drv = torch.tensor([(y_drv[tr_idx] == 0).sum() / (y_drv[tr_idx] == 1).sum()], device=device)

@torch.no_grad()
def evaluate(ei_t1, y_t1, w_t1, idx_t2):
    model.eval()
    h = model.encode_nodes()

    # Task 1 overall + Hard-Negative-only (Causality=Yes vs Causality=No)
    p1 = torch.sigmoid(model.decode_links(h["lncrna"], h["disease"], ei_t1)).cpu().numpy()
    y1 = y_t1.cpu().numpy()
    w1 = w_t1.cpu().numpy()
    auc_all = roc_auc_score(y1, p1)
    hard_mask = w1 == 1.0  # isolates Causality=Yes (y=1) vs Causality=No (y=0)
    auc_hard = roc_auc_score(y1[hard_mask], p1[hard_mask])

    # Task 2 Glioma driver
    p2 = torch.sigmoid(model.driver_head(h["lncrna"][idx_t2]).squeeze(-1)).cpu().numpy()
    y2 = y_drv[idx_t2].cpu().numpy()
    auc_drv = roc_auc_score(y2, p2)
    ap_drv = average_precision_score(y2, p2)
    return auc_all, auc_hard, auc_drv, ap_drv

best_val = -1.0
for epoch in range(1, 61):
    model.train()
    optimizer.zero_grad()

    h = model.encode_nodes()

    # Task 1: Pan-disease causal link loss
    logits_t1 = model.decode_links(h["lncrna"], h["disease"], t1_tr_ei)
    loss_t1 = F.binary_cross_entropy_with_logits(logits_t1, t1_tr_y, weight=t1_tr_w)

    # Task 2a: Glioma causal driver loss
    logits_drv = model.driver_head(h["lncrna"][tr_idx]).squeeze(-1)
    loss_drv = F.binary_cross_entropy_with_logits(logits_drv, y_drv[tr_idx], pos_weight=pos_weight_drv)

    # Task 2b: Oncogenic vs tumor-suppressive polarity loss
    pol_mask = tr_idx[y_dir[tr_idx] >= 0]
    logits_pol = model.polarity_head(h["lncrna"][pol_mask]).squeeze(-1)
    loss_pol = F.binary_cross_entropy_with_logits(logits_pol, y_dir[pol_mask])

    loss = loss_t1 + loss_drv + 0.5 * loss_pol
    loss.backward()
    optimizer.step()

    if epoch % 5 == 0:
        va_all, va_hard, va_drv, va_ap = evaluate(t1_va_ei, t1_va_y, t1_va_w, va_idx)
        score = 0.5 * va_hard + 0.5 * va_drv
        if score > best_val:
            best_val = score
            torch.save(model.state_dict(), os.path.join(OUT_DIR, "multitask_causal_gnn_best.pt"))
        print(
            f"Ep {epoch:02d} | Loss {loss.item():.4f} | "
            f"Val Link AUROC {va_all:.4f} (Causal-vs-No {va_hard:.4f}) | "
            f"Val Glioma Driver AUROC {va_drv:.4f} / AUPRC {va_ap:.4f}"
        )

model.load_state_dict(torch.load(os.path.join(OUT_DIR, "multitask_causal_gnn_best.pt")))
te_all, te_hard, te_drv, te_ap = evaluate(t1_te_ei, t1_te_y, t1_te_w, te_idx)
print(
    f"\n=== TEST RESULTS ===\n"
    f"Pan-Disease Causal Link AUROC : {te_all:.4f}\n"
    f"Hard Causal-vs-NonCausal AUROC: {te_hard:.4f}\n"
    f"Glioma Causal Driver AUROC    : {te_drv:.4f} | AUPRC: {te_ap:.4f}"
)
