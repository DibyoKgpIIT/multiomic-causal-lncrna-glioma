import os
import cptac
import decoupler as dc
import pandas as pd

os.makedirs("data/raw", exist_ok=True)

# 1. Fetch curated human TF-target causal network via decoupler (or OmniPath with genesymbols=1)
tf_net = dc.op.collectri(organism="human")
tf_net.to_csv("data/raw/collectri_human.csv", index=False)
print(f"Saved CollecTRI: {len(tf_net)} edges, {tf_net['source'].nunique()} unique TFs")

# 2. Download Brain (GBM) matched transcriptomics and proteomics (cptac >= 1.5 auto-downloads)
gbm = cptac.Gbm()
rna_df = gbm.get_transcriptomics(source="bcm")
prot_df = gbm.get_proteomics(source="bcm")

# Flatten MultiIndex columns if present (keep HGNC gene symbol on level 0)
if isinstance(rna_df.columns, pd.MultiIndex):
    rna_df.columns = rna_df.columns.get_level_values(0)
if isinstance(prot_df.columns, pd.MultiIndex):
    prot_df.columns = prot_df.columns.get_level_values(0)

# Deduplicate columns by averaging duplicate isoform/gene symbol columns if any
rna_df = rna_df.T.groupby(level=0).mean().T
prot_df = prot_df.T.groupby(level=0).mean().T

# 3. Align samples with both RNA-seq and Proteomics, and filter proteomics to active loop TFs
common_samples = rna_df.index.intersection(prot_df.index)
rna_df = rna_df.loc[common_samples]
active_tfs = sorted(list(set(tf_net["source"]).intersection(set(prot_df.columns))))
tf_prot_df = prot_df.loc[common_samples, active_tfs]

rna_df.to_parquet("data/raw/gbm_rna.parquet")
tf_prot_df.to_parquet("data/raw/gbm_tf_prot.parquet")

print(f"Matched GBM samples: {len(common_samples)}")
print(f"RNA features (mRNA + lncRNA): {rna_df.shape[1]}")
print(f"Active loop TFs quantified: {tf_prot_df.shape[1]}")
