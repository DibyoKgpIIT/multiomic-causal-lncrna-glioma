import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, Linear
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
OUT_DIR = os.path.join(BASE_DIR, "data/processed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
base_data = bundle["graph"]
splits = torch.load(os.path.join(OUT_DIR, "multitask_splits.pt"), weights_only=False)

for et in list(base_data.edge_types):
    if base_data[et].edge_index.numel() == 0:
        del base_data[et]

valid_lnc_mask = base_data["lncrna"].strict_lnc_mask
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

# Strict entity-disjoint Task 1 training edges
heldout_lncs = set(inner_va_idx.tolist()) | set(val_idx.tolist()) | set(test_idx.tolist())
t1_tr_ei, t1_tr_y, t1_tr_w = splits["task1"]["train"]
keep_t1_tr = torch.tensor([int(u) not in heldout_lncs for u in t1_tr_ei[0]], dtype=torch.bool)
t1_tr_ei, t1_tr_y, t1_tr_w = t1_tr_ei[:, keep_t1_tr].to(device), t1_tr_y[keep_t1_tr].to(device), t1_tr_w[keep_t1_tr].to(device)

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


class MultiOmicCausalGNN(nn.Module):
    def __init__(self, data_obj, hidden_dim=64, dropout=0.35, use_edge_signs=True):
        super().__init__()
        self.dropout = dropout
        self.use_edge_signs = use_edge_signs
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
            conv_dict = {}
            for et in data_obj.edge_types:
                if use_edge_signs and "regulates" in et[1] and "edge_attr" in data_obj[et]:
                    conv_dict[et] = GATv2Conv((-1, -1), hidden_dim, edge_dim=1, add_self_loops=False)
                else:
                    conv_dict[et] = SAGEConv((-1, -1), hidden_dim)
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

        edge_attr_dict = (
            {et: data_obj[et].edge_attr for et in data_obj.edge_types if "edge_attr" in data_obj[et]}
            if self.use_edge_signs else {}
        )
        for conv, norm in zip(self.convs, self.norms):
            h_next = conv(h, data_obj.edge_index_dict, edge_attr_dict=edge_attr_dict) if edge_attr_dict else conv(h, data_obj.edge_index_dict)
            for nt in h:
                if nt in h_next:
                    h[nt] = norm[nt](h[nt] + F.dropout(F.gelu(h_next[nt]), p=self.dropout, training=self.training))
        return h

    def decode_links(self, h_lnc, h_dis, edge_index):
        src = h_lnc[edge_index[0]]
        dst = h_dis[edge_index[1]]
        return self.link_head(torch.cat([src, dst, src * dst], dim=-1)).squeeze(-1)


def run_experiment(name, single_task=False, drop_cptac_proteomics=False, drop_edge_signs=False):
    torch.manual_seed(42)
    g = base_data.clone()
    if drop_cptac_proteomics:
        # Keep only first 12 dims (GTEx Normal Brain + TCGA-LGG/GBM cohort stats); strip CPTAC-GBM proteomics & TF coupling
        g["lncrna"].x = g["lncrna"].x[:, :12]
        g["mrna"].x = g["mrna"].x[:, :12]
        g["protein"].x = g["protein"].x[:, :13]
    g = g.to(device)

    model = MultiOmicCausalGNN(g, hidden_dim=64, dropout=0.35, use_edge_signs=(not drop_edge_signs)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=1e-3)

    best_score = -1.0
    best_state = None
    best_ep = 0

    for ep in range(1, 41):
        model.train()
        opt.zero_grad()
        h = model.encode_nodes(g)

        loss_drv = F.binary_cross_entropy_with_logits(
            model.driver_head(h["lncrna"][inner_tr_d]).squeeze(-1), y_drv_d[inner_tr_d], pos_weight=pos_weight_drv
        )
        loss_pol = F.binary_cross_entropy_with_logits(
            model.polarity_head(h["lncrna"][pol_tr]).squeeze(-1), y_dir_d[pol_tr], pos_weight=pos_weight_pol
        )
        if single_task:
            loss = 1.4 * loss_drv + 0.5 * loss_pol
        else:
            loss_t1 = F.binary_cross_entropy_with_logits(
                model.decode_links(h["lncrna"], h["disease"], t1_tr_ei), t1_tr_y, weight=t1_tr_w
            )
            loss = 0.6 * loss_t1 + 1.4 * loss_drv + 0.5 * loss_pol

        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            h_eval = model.encode_nodes(g)
            p_drv = torch.sigmoid(model.driver_head(h_eval["lncrna"][inner_va_d]).squeeze(-1)).cpu().numpy()
            auc_drv = roc_auc_score(y_drv_d[inner_va_d].cpu().numpy(), p_drv)
            if single_task:
                score = auc_drv
            else:
                p1 = torch.sigmoid(model.decode_links(h_eval["lncrna"], h_eval["disease"], t1_va_ei)).cpu().numpy()
                y1, w1 = t1_va_y.cpu().numpy(), t1_va_w.cpu().numpy()
                auc_hard = roc_auc_score(y1[w1 == 1.0], p1[w1 == 1.0])
                score = 2 * auc_hard * auc_drv / (auc_hard + auc_drv + 1e-8)

            if score > best_score:
                best_score = score
                best_ep = ep
                best_state = {k: v.cpu() for k, v in model.state_dict().items()}

    model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
    model.eval()
    with torch.no_grad():
        h_te = model.encode_nodes(g)
        p2 = torch.sigmoid(model.driver_head(h_te["lncrna"][test_d]).squeeze(-1)).cpu().numpy()
        y2 = y_drv_d[test_d].cpu().numpy()
        te_drv_auc = roc_auc_score(y2, p2)
        te_drv_ap = average_precision_score(y2, p2)

        if single_task:
            te_link_auc, te_hard_auc = float("nan"), float("nan")
        else:
            p1 = torch.sigmoid(model.decode_links(h_te["lncrna"], h_te["disease"], t1_te_ei)).cpu().numpy()
            y1, w1 = t1_te_y.cpu().numpy(), t1_te_w.cpu().numpy()
            te_link_auc = roc_auc_score(y1, p1)
            te_hard_auc = roc_auc_score(y1[w1 == 1.0], p1[w1 == 1.0])

    return {
        "Variant": name,
        "Best_Ep": best_ep,
        "Test_Link_AUROC": te_link_auc,
        "Test_Hard_Causal_AUROC": te_hard_auc,
        "Test_Glioma_Driver_AUROC": te_drv_auc,
        "Test_Glioma_Driver_AUPRC": te_drv_ap,
    }


results = [
    run_experiment("Full Multi-Task Multi-Omic GNN", single_task=False, drop_cptac_proteomics=False, drop_edge_signs=False),
    run_experiment("Single-Task (No Pan-Disease)", single_task=True, drop_cptac_proteomics=False, drop_edge_signs=False),
    run_experiment("No CPTAC-GBM TF Proteomics", single_task=False, drop_cptac_proteomics=True, drop_edge_signs=False),
    run_experiment("Unsigned CollecTRI Edges", single_task=False, drop_cptac_proteomics=False, drop_edge_signs=True),
]

df_abl = pd.DataFrame(results)
df_abl.to_csv(os.path.join(OUT_DIR, "ablation_results.csv"), index=False)
print("\n=== LEAK-FREE ABLATION BENCHMARK (HELD-OUT TEST SET) ===")
print(df_abl.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
