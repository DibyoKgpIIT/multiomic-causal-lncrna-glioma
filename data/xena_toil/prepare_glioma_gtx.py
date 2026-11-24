import pandas as pd
import numpy as np

# 1. Load phenotype table and normalize column names
pheno = pd.read_csv("TcgaTargetGTEX_phenotype.txt.gz", sep="\t", encoding="latin1")
pheno.columns = [c.strip() for c in pheno.columns]
print("Phenotype columns:", pheno.columns.tolist())

disease_col = "primary disease or tissue"

is_gtex_brain = (pheno["_study"] == "GTEX") & (pheno["_primary_site"] == "Brain")
is_tcga_glioma = (pheno["_study"] == "TCGA") & (
    pheno[disease_col].str.lower().isin(["glioblastoma multiforme", "brain lower grade glioma"])
)

sub_pheno = pheno[is_gtex_brain | is_tcga_glioma].copy()
print("\nCohort breakdown:\n", sub_pheno.groupby(["_study", disease_col, "_sample_type"]).size())

# 2. Match sample IDs against TPM matrix header
tpm_path = "TcgaTargetGtex_rsem_gene_tpm.gz"
all_cols = pd.read_csv(tpm_path, sep="\t", nrows=0).columns.tolist()

valid_samples = set(sub_pheno["sample"])
keep_samples = [c for c in all_cols[1:] if c in valid_samples]
print(f"\nMatched {len(keep_samples)} / {len(sub_pheno)} samples in TPM matrix.")

# 3. Load only matched columns
df_expr = pd.read_csv(tpm_path, sep="\t", usecols=["sample"] + keep_samples)
df_expr["ensembl_id"] = df_expr["sample"].str.split(".").str[0]

# 4. Convert from log2(TPM + 0.001) to log2(TPM + 1)
tpm_linear = np.maximum(np.power(2.0, df_expr[keep_samples].values) - 0.001, 0.0)
df_expr[keep_samples] = np.log2(tpm_linear + 1.0)

# 5. Save outputs
sub_pheno[sub_pheno["sample"].isin(keep_samples)].to_csv("glioma_gtex_phenotype.csv", index=False)
df_expr.to_parquet("glioma_gtex_log2tpm1.parquet", index=False)
print("Saved expression shape:", df_expr.shape)
