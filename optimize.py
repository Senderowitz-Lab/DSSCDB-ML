import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True, stream=sys.stdout)

import optuna
optuna.logging.set_verbosity(optuna.logging.INFO)

from optuna_objectives import *
from gnn_model import precompute_graphs
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import euclidean_distances
import joblib
from functools import partial

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
N_TRIALS = 50
N_JOBS_OPTUNA = 1
OUT_DIR = 'Output'
IMPUTE_METHODS = ("mean", "median", "most_frequent", None)
MODEL_FAMILIES = ('hgb', 'rf', 'mlp')
TEST_SIZE = 0.2

os.makedirs(OUT_DIR, exist_ok=True)

# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
smiles_df = pd.read_csv("Data/cleaned_SMILES_fixed.csv", index_col=0)
dye_device_descs = pd.read_csv("Data/dye_device_pce_without_dupes_new.csv", index_col=0)
pce = dye_device_descs['PCE'].dropna()
dye_device_descs = dye_device_descs.drop('PCE', axis=1).copy().loc[pce.index]
smiles_df = smiles_df.loc[dye_device_descs.index]

print(f'Loaded {len(dye_device_descs)} lines')

fp_cols = [c for c in dye_device_descs.columns if c.startswith("Bit_")]
device_cols = [c for c in dye_device_descs.columns if not c.startswith("Bit_")]

print(f"{len(fp_cols)} fingerprint columns, {len(device_cols)} device features")

graph_cache = precompute_graphs(smiles_df["SMILES"])
print(f"Cached {len(graph_cache)} unique molecular graphs")

groups = smiles_df.loc[dye_device_descs.index, 'SMILES']

# ──────────────────────────────────────────────
# Kennard-Stone group-based split
# ──────────────────────────────────────────────
def kennard_stone_groups(X_df, groups, fp_cols, test_size=0.2):
    """
    Kennard-Stone selection at the group (dye) level.
    Distances computed on Morgan fingerprints only.
    Selects dyes until ~test_size fraction of total samples is reached.
    """
    unique_dyes = groups.unique()
    n_total = len(groups)
    n_test_target = int(n_total * test_size)
 
    dye_reps = []
    dye_sizes = {}
    for dye in unique_dyes:
        first_idx = groups[groups == dye].index[0]
        dye_reps.append(X_df.loc[first_idx, fp_cols].values)
        dye_sizes[dye] = (groups == dye).sum()
    dye_reps = np.array(dye_reps)
 
    dist_matrix = euclidean_distances(dye_reps)
 
    i, j = np.unravel_index(dist_matrix.argmax(), dist_matrix.shape)
    selected = [i, j]
    n_test_samples = dye_sizes[unique_dyes[i]] + dye_sizes[unique_dyes[j]]
 
    while n_test_samples < n_test_target:
        remaining = [k for k in range(len(unique_dyes)) if k not in selected]
        if not remaining:
            break
        min_dists = dist_matrix[np.ix_(remaining, selected)].min(axis=1)
        best = remaining[np.argmax(min_dists)]
        selected.append(best)
        n_test_samples += dye_sizes[unique_dyes[best]]
 
    test_dyes = set(unique_dyes[selected])
    train_dyes = set(unique_dyes) - test_dyes
 
    test_idx = groups[groups.isin(test_dyes)].index
    train_idx = groups[groups.isin(train_dyes)].index
 
    return train_idx, test_idx, train_dyes, test_dyes


# ──────────────────────────────────────────────
# Build KS split for optimization
# ──────────────────────────────────────────────
train_idx, test_idx, train_dyes, test_dyes = kennard_stone_groups(
    dye_device_descs, groups, fp_cols, test_size=TEST_SIZE
)
 
n_total = len(dye_device_descs)
print(f"\nKennard-Stone split:")
print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_total:.1%}), "
      f"{len(train_dyes)} unique dyes")
print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/n_total:.1%}), "
      f"{len(test_dyes)} unique dyes")
assert len(train_dyes & test_dyes) == 0, "Dye leakage detected!"
print("Split is leak-free.\n")

# ──────────────────────────────────────────────
# Family map
# ──────────────────────────────────────────────
study_gnn = partial(optuna_gnn_study,
                    smiles_series=smiles_df["SMILES"],
                    graph_cache=graph_cache)

family_map = {
    "mlp": (optuna_mlp_study, "mlp"),
    "rf":  (optuna_rf_study,  "rf"),
    "hgb": (optuna_hgb_study, "hgb"),
    "gnn": (study_gnn,        "gnn"),
}

# ──────────────────────────────────────────────
# Run optimization on KS training set
# ──────────────────────────────────────────────
best_params_all = {}
 
for fam_name in MODEL_FAMILIES:
    if fam_name not in family_map:
        raise ValueError(f"Unknown family '{fam_name}'")
 
    study_fn, fam_tag = family_map[fam_name]
 
    if fam_tag == "gnn":
        X0 = dye_device_descs.loc[train_idx, device_cols]
    else:
        X0 = dye_device_descs.loc[train_idx]
 
    y0 = pce.loc[train_idx]
    g0 = groups.loc[train_idx]
 
    for impute_method in IMPUTE_METHODS:
        key = f"{fam_tag}_{impute_method}"
        impute_label = impute_method if impute_method is not None else "KNN_imputation"
        print(f"\n{'='*60}")
        print(f"Optimizing: {fam_tag} + {impute_label}")
        print(f"{'='*60}")
 
        best_params = study_fn(
            X0, y0, g0,
            n_trials=N_TRIALS,
            n_jobs=N_JOBS_OPTUNA,
            impute_strategy=impute_method,
        )
        best_params = getattr(best_params, "best_params", best_params)
        best_params_all[key] = best_params
 
        print(f"Best params for {key}: {best_params}")
 
# ──────────────────────────────────────────────
# Save
# ──────────────────────────────────────────────
params_path = os.path.join(OUT_DIR, "best_params_ks.joblib")
joblib.dump(best_params_all, params_path)
print(f"\nSaved {len(best_params_all)} param sets to {params_path}")
 
with open(os.path.join(OUT_DIR, "best_params_ks.txt"), 'w') as f:
    for key, params in best_params_all.items():
        f.write(f"{key}:\n")
        for k, v in params.items():
            f.write(f"  {k}: {v}\n")
        f.write("\n")
 
print("KS-based optimization complete.")