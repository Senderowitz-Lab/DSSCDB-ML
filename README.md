# Global QSAR Model for Dye-Sensitized Solar Cell PCE Prediction

**Combining Device Information and Molecular Fingerprints into a Global Model for Reliable Prediction of Power Conversion Efficiency of Dye Sensitized Solar Cells**

Ana María Juárez Marckwordt, Paul F. A. Clarke, Hanoch Senderowitz*

Department of Chemistry, Bar-Ilan University, Ramat-Gan, 5290002, Israel  
*Corresponding author: hsenderowitz@gmail.com

---

## Overview

This repository contains the code, data, and results for a global machine learning (ML) model that predicts the Power Conversion Efficiency (PCE) of dye-sensitized solar cells (DSSCs). The model is trained on dye-device pairs from the DSSC Database (DSSCDB), combining Morgan fingerprints (ECFP6) of dye molecules with extracted device features.

The best-performing model is a Histogram-based Gradient Boosting (HGB) regressor with mean imputation, achieving **Q² = 0.73 ± 0.02, MAE = 0.97 ± 0.03** across 100 random group-based splits.

---

## Repository Structure

```
├── Data/
│   ├── cleaned_SMILES.csv                    # Cleaned dye SMILES 
│   ├── cleaned_SMILES_fixed.csv              # Cleaned dye SMILES (Ru complexes corrected)
│   ├── dye_device_pce_without_dupes_new.csv  # Dye-device pairs with PCE
│   └── rawDSSCDBdata.csv                     # Raw DSSC Database (Venkatraman, et al.)
├── Output/
│   ├── best_params_ks.joblib             # Optimized hyperparameters (KS split) (HGB, RF, MLP)
│   ├── best_params_ks.txt                # Human-readable hyperparameters (HGB, RF, MLP)
│   ├── metrics_summary.csv               # Performance metrics (100 splits) (HGB, RF, MLP)
│   ├── best_params_ks_gnn.joblib         # Optimized hyperparameters (KS split) (GNN)
│   ├── best_params_ks_gnn.txt            # Human-readable hyperparameters (GNN)
│   ├── metrics_summary_gnn.csv           # Performance metrics (100 splits) (GNN)
│   ├── splits_indices.joblib             # 100 random splits (HGB, RF, MLP)
│   ├── splits_indices_gnn.joblib         # 10 random splits (GNN)
│   └── SHAP/
│       ├── feature_importance_shap.csv   # SHAP feature importances
│       ├── beeswarm_overall.png          # Overall SHAP summary plot
│       ├── beeswarm_triphenylamine.png   # SHAP plot for triarylamine family
│       ├── beeswarm_phenothiazine.png    # SHAP plot for phenothiazine family
│       ├── beeswarm_carbazole.png        # SHAP plot for carbazole family
│       ├── beeswarm_low.png              # SHAP plot for low PCE range
│       ├── beeswarm_medium.png           # SHAP plot for medium PCE range
│       ├── beeswarm_high.png             # SHAP plot for high PCE range
│       └── dependence_top6.png           # Dependence plots for top 6 features
├── requirements.txt                      # Dependencies
├── data_prep.ipynb                       # Data cleaning and feature extraction
├── cheminfo_functions.py                 # Chemoinformatics utilities
├── model_explain_functions.py            # SHAP visualization utilities
├── extras.ipynb                          # Contains utilities for ad-hoc analysis (not required to reproduce the main results)
├── optuna_objectives.py                  # Optuna objective functions for all algorithms
├── optimize.py                           # Hyperparameter optimization on KS split
├── train_splits.py                       # Train and evaluate across 100 random splits
├── gnn_model.py                          # GATv2Conv GNN architecture
└── SHAP.py                               # SHAP analysis and visualization
```

---

## Requirements

Install dependencies:
```bash
pip install -r requirements.txt
```

For RDKit (recommended via conda):
```bash
conda install -c conda-forge rdkit==2025.3.3
```

Note: PyTorch installation may vary by platform and CUDA version.  
See https://pytorch.org/get-started/locally/ for the correct command for your system.

---

## Usage

### 1. Data Preparation
Open and run `data_prep.ipynb` to reproduce the cleaned dataset from the raw DSSCDB.

### 2. Hyperparameter Optimization
```bash
python optimize.py
```
Runs Optuna optimization for HGB, RF, and MLP across four imputation methods (12 combinations) on the Kennard-Stone split. Saves results to `Output/best_params_ks.joblib`.

Due to the significantly longer training time of the GNN, its optimization is run separately:
```bash
python optimize.py  # set MODEL_FAMILIES = ('gnn',) in configuration
```
Saves GNN results to `Output/best_params_ks_gnn.joblib`.

### 3. Train and Evaluate
```bash
python train_splits.py
```
Trains HGB, RF, and MLP across 100 random group-based splits. Saves metrics to `Output/metrics_summary.csv`.

Due to the long training time of GNN, it is evaluated on 10 splits separately by setting `MODEL_FAMILIES = ('gnn',)` and `N_SPLITS = 10` in `train_splits.py`. Saves metrics to `Output/metrics_summary_gnn.csv`.

### 4. SHAP Analysis
```bash
python SHAP.py
```
Generates all SHAP summary plots for the best model (HGB + mean imputation) on the Kennard-Stone split, including overall, per-family, and per-PCE-range analyses. Saves figures to `Output/SHAP/`.

---

## Key Features

- **Global model**: Covers all 4,351 entries in the DSSCDB, including organic, metal-organic, and dye mixture entries
- **Group-based splitting**: All splits enforce strict dye-level separation between training and test sets to prevent data leakage
- **Kennard-Stone split**: Deterministic split based on ECFP6 Euclidean distances ensures maximum chemical diversity in the test set
- **Four ML algorithms**: HGB, RF, MLP, and GNN evaluated across four imputation strategies (16 combinations)
- **SHAP interpretation**: Feature importances computed with TreeExplainer for chemically meaningful interpretation

---

## Citation

If you use this code or data, please cite:

> Juárez Marckwordt, A. M., Clarke, P. F. A., & Senderowitz, H. (2026). Combining Device Information and Molecular Fingerprints into a Global Model for Reliable Prediction of Power Conversion Efficiency of Dye-Sensitized Solar Cells. *J. Chem. Inf. Model.* https://doi.org/10.1021/acs.jcim.6c01826

---

## License

This project is licensed under the MIT License.
