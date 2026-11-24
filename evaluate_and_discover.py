import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import HeteroConv, SAGEConv, GATv2Conv, Linear
import pandas as pd

BASE_DIR = "/media/experiment/ssd_1TB/multiomic-causal"
OUT_DIR = os.path.join(BASE_DIR, "data/processed")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

bundle = torch.load(os.path.join(OUT_DIR, "multiomic_causal_hetero_full.pt"), weights_only=False)
data = bundle["graph"]
node_maps = bundle["node_maps"]
splits = torch.load(os.path.join(OUT_DIR, "multitask_splits.pt"), weights_only=False)

for et in list(data.edge_types):
    if data[et].edge_index.numel() == 0:
        del data[et]

data = data.to(device)

class MultiOmicCausalGNN(nn.Module):
    def __init__(self, metadata, hidden_dim=96, dropout=0.25):
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

model = MultiOmicCausalGNN(data.metadata(), hidden_dim=96).to(device)
model.load_state_dict(torch.load(os.path.join(OUT_DIR, "multitask_causal_gnn_best.pt"), weights_only=True))
model.eval()

idx_to_lnc = {v: k for k, v in node_maps["lncrna"].items()}
idx_to_dis = {v: k for k, v in node_maps["disease"].items()}

glioma_dis_ids = [idx for name, idx in node_maps["disease"].items() if "glioma" in name or "glioblastoma" in name]
print("Matched glioma disease nodes:", [idx_to_dis[i] for i in glioma_dis_ids])

with torch.no_grad():
    h = model.encode_nodes()
    h_lnc = h["lncrna"]
    n_lnc = h_lnc.size(0)

    p_driver = torch.sigmoid(model.driver_head(h_lnc).squeeze(-1)).cpu()
    p_polarity = torch.sigmoid(model.polarity_head(h_lnc).squeeze(-1)).cpu()

    p_links = []
    for d_id in glioma_dis_ids:
        ei = torch.stack([torch.arange(n_lnc, device=device), torch.full((n_lnc,), d_id, device=device)], dim=0)
        p_links.append(torch.sigmoid(model.decode_links(h_lnc, h["disease"], ei)).cpu())
    p_glioma_link = torch.stack(p_links, dim=0).max(dim=0).values

y_drv = splits["task2"]["y_driver"]

rows = []
for i in range(n_lnc):
    status = "Known_Glioma_Driver" if y_drv[i] == 1 else ("NonGlioma_Causal" if y_drv[i] == 0 else "Unannotated")
    rows.append({
        "symbol": idx_to_lnc[i],
        "status": status,
        "driver_prob": float(p_driver[i]),
        "glioma_link_prob": float(p_glioma_link[i]),
        "joint_causal_score": float(0.5 * p_driver[i] + 0.5 * p_glioma_link[i]),
        "oncogenic_prob": float(p_polarity[i]),
        "predicted_role": "Oncogenic (Up)" if p_polarity[i] >= 0.5 else "Suppressive (Down)",
    })

df_res = pd.DataFrame(rows).sort_values("joint_causal_score", ascending=False)
df_res.to_csv(os.path.join(OUT_DIR, "glioma_causal_predictions.csv"), index=False)

print("\n=== TOP 20 NOVEL (UNANNOTATED) CAUSAL GLIOMA CANDIDATES ===")
novel = df_res[df_res["status"] == "Unannotated"].head(20)
print(novel[["symbol", "joint_causal_score", "driver_prob", "glioma_link_prob", "oncogenic_prob", "predicted_role"]].to_string(index=False))

print("\n=== TOP 10 HELD-OUT TEST GLIOMA DRIVERS ===")
test_pos = [int(i) for i in splits["task2"]["test_idx"] if y_drv[i] == 1]
print(df_res.set_index("symbol").loc[[idx_to_lnc[i] for i in test_pos]].sort_values("joint_causal_score", ascending=False)[
    ["joint_causal_score", "driver_prob", "glioma_link_prob", "predicted_role"]
].head(10).to_string())
