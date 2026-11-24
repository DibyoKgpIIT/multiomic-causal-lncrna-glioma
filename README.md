# Multi-Task Multi-Omic Causal Heterogeneous GNN for Oncogenic lncRNA Prioritization in Glioma

Official implementation and reproducible benchmark suite for **"Multi-Task Multi-Omic Causal Heterogeneous Graph Neural Networks for Predicting Oncogenic Long Non-Coding RNA Drivers in Glioma"** (*ICBBB 2026*).

## Overview

Distinguishing upstream causal long non-coding RNAs (lncRNAs) from downstream co-expressed passenger transcripts or immune-infiltrated markers remains a major bottleneck in diffuse glioma and glioblastoma (GBM) genomics. This repository implements an **inductive, entity-disjoint multi-task heterogeneous graph neural network (GNN)** integrating:

1. **Normal-to-Tumor Progression Trajectories:** Batch-harmonized UCSC Xena TOIL expression ($\log_2(\text{TPM}+1)$) across GTEx normal brain, TCGA-LGG, and TCGA-GBM (`12D` cohort statistics and pairwise log-fold changes).
2. **Patient-Matched GBM TF Proteomics:** Paired RNA-seq and mass-spectrometry transcription factor abundance ($n = 99$ CPTAC-GBM patients, $684$ TF proteins) capturing empirical post-transcriptional TF coupling (`9D`).
3. **Multi-Modal Regulatory Interactome:** Heterogeneous graph ($56,823$ nodes across `lncRNA`, `mRNA`, `Protein/TF`, `miRNA`, `Disease`, and `Drug`) combining RNAInter v4.0, CollecTRI regulons, disease comorbidity links, and ChG-Miner drug–target interactions.
4. **Strict Biotype & Anti-Leakage Hygiene:** HGNC biotype filtering ($5,611$ verified non-coding lncRNAs) and a strictly inductive `70% / 15% / 15%` entity-disjoint split where all disease links incident to held-out lncRNAs are masked prior to message passing.

---

## Repository Structure

```text
multiomic-causal/
├── build_hetero_graph.py              # Base heterogeneous graph assembly (RNAInter, LncRNADisease, ChG-Miner)
├── fix_drug_nodes.py                  # Drug node harmonization and DrugBank/ChG-Miner mapping
├── enrich_hetero_graph.py             # Integration of CollecTRI TF-target regulons and miRNA interactions
├── upgrade_causal_features.py         # Multi-omic node feature engineering (GTEx/TCGA + CPTAC-GBM TF coupling)
├── prepare_multitask_labels.py        # HGNC biotype filtering & entity-disjoint 70/15/15 split generation
├── train_unsigned_causal.py           # Finalized multi-task unsigned SAGEConv GNN (d=64)
├── evaluate_and_discover.py           # Held-out test evaluation & novel glioma candidate prioritization
├── explain_candidates.py              # 1-hop TF/RBP & cis-regulatory subgraph extraction
├── explain_2hop_pathways.py           # 2-hop ceRNA (lncRNA -> miRNA -> mRNA) mechanistic cascades
├── run_ablations.py                   # Controlled ablations (Single-task, Signed GATv2Conv, No Proteomics, d=96)
├── run_clean_benchmark.py             # Clean benchmark verification across inductive splits
├── sequence_embedding_ablation3.py    # FFT-based SWSPM sequence spectrum ablation (390D & 16D SVD)
└── data/
    ├── processed/
    │   ├── glioma_causal_predictions_clean.csv      # Ranked predictions across 5,611 HGNC-verified lncRNAs
    │   ├── ablation_results.csv                     # Architectural ablation benchmark results
    │   └── swspm_sequence_ablation_results.csv      # FFT-SWSPM sequence embedding ablation results
    └── xena_toil/
        └── prepare_glioma_gtx.py                    # Cohort preprocessing for GTEx Brain / TCGA-LGG / TCGA-GBM
```

---

## Key Results (Entity-Disjoint Held-Out Test Set)

| Model Variant | Link AUROC | Hard Causal-vs-No AUROC | Glioma Driver AUROC | Glioma Driver AUPRC |
| :--- | :---: | :---: | :---: | :---: |
| Single-Task (Glioma Only) | -- | -- | 0.6651 | 0.6070 |
| Signed CollecTRI (`GATv2Conv`) | 0.9485 | 0.7998 | 0.6181 | 0.5524 |
| No CPTAC-GBM Proteomics | 0.9522 | 0.8070 | 0.6490 | 0.5476 |
| Raw Expr. Skip Connection | 0.9508 | 0.8088 | 0.6077 | 0.5193 |
| + Raw FFT-SWSPM Seq. (390D) | 0.9489 | 0.8074 | 0.5700 | 0.4703 |
| + SVD FFT-SWSPM Seq. (16D) | 0.9508 | 0.8129 | 0.5987 | 0.5065 |
| Unsigned Multi-Task ($d=96$) | 0.9507 | 0.7990 | 0.6442 | 0.5634 |
| **Finalized Unsigned ($d=64$)** | **0.9517** | **0.8056** | **0.6714** | **0.5960** |

---

## Setup & Reproduction

### 1. Environment Installation
```bash
conda create -n causal_graph python=3.11 -y
conda activate causal_graph
pip install torch torch_geometric numpy pandas scikit-learn scipy pyarrow
```

### 2. Data Preparation & Graph Construction
Place raw source files from **UCSC Xena TOIL** (`TcgaTargetGtex_rsem_gene_tpm.gz`, `TcgaTargetGTEX_phenotype.txt.gz`), **CPTAC-GBM** (`gbm_rna.parquet`, `gbm_tf_prot.parquet`), **CollecTRI** (`collectri_human.csv`), **HGNC** (`hgnc_complete_set.txt`), **RNAInter v4.0**, **LncRNADisease v3.0**, **LncRNAWiki 2.0**, and **ChG-Miner** under `data/`, then run:

```bash
python data/xena_toil/prepare_glioma_gtx.py
python build_hetero_graph.py
python fix_drug_nodes.py
python enrich_hetero_graph.py
python upgrade_causal_features.py
python prepare_multitask_labels.py
```

### 3. Training, Evaluation & Mechanistic Interpretation
```bash
# Train the finalized d=64 unsigned multi-task causal GNN
python train_unsigned_causal.py

# Evaluate on the held-out test set and generate ranked predictions
python evaluate_and_discover.py

# Extract 1-hop TF/RBP hubs and 2-hop ceRNA regulatory cascades
python explain_candidates.py
python explain_2hop_pathways.py

# Run controlled architectural and FFT-SWSPM sequence ablations
python run_ablations.py
python sequence_embedding_ablation3.py
```

---

## Citation

If you use this code or prioritized candidate list in your research, please cite:

```bibtex
@inproceedings{haldar2026multiomic,
  title={Multi-Task Multi-Omic Causal Heterogeneous Graph Neural Networks for Predicting Oncogenic Long Non-Coding RNA Drivers in Glioma},
  author={Haldar, Dibya Kanti and Kishore, Hrishabh and Mitra, Pralay},
  booktitle={Proceedings of the International Conference on Bioscience, Biochemistry and Bioinformatics (ICBBB)},
  year={2026}
}
```