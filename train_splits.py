import os
import sys

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

import logging
for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(level=logging.INFO, format='%(message)s', force=True, stream=sys.stdout)

from optuna_objectives import *
from gnn_model import precompute_graphs
import pandas as pd
import numpy as np
import joblib
from functools import partial

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
N_SPLITS = 100
OUT_DIR = 'Output'
IMPUTE_METHODS = ("mean", "median", "most_frequent", None,)
MODEL_FAMILIES = ("hgb", "rf", "mlp",)
SAVE_MODELS_ONLY_I0 = True
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
# Random group-based split (20% of samples)
# ──────────────────────────────────────────────
def random_group_split(groups, test_size=0.2, random_state=42):
    """
    Random group-based split targeting ~20% of total samples in test.
    No dye appears in both train and test.
    """
    rng = np.random.RandomState(random_state)
    unique_dyes = groups.unique()
    n_total = len(groups)
    n_test_target = int(n_total * test_size)

    dye_sizes = {dye: (groups == dye).sum() for dye in unique_dyes}

    shuffled = rng.permutation(unique_dyes)

    test_dyes = set()
    n_test_samples = 0
    for dye in shuffled:
        if n_test_samples >= n_test_target:
            break
        test_dyes.add(dye)
        n_test_samples += dye_sizes[dye]

    train_dyes = set(unique_dyes) - test_dyes

    test_idx = groups[groups.isin(test_dyes)].index
    train_idx = groups[groups.isin(train_dyes)].index

    return train_idx, test_idx


# ──────────────────────────────────────────────
# Build N splits with different random seeds
# ──────────────────────────────────────────────
idx_all = dye_device_descs.index
n_total = len(idx_all)
splits = []

for i in range(N_SPLITS):
    tr_idx, te_idx = random_group_split(groups, test_size=TEST_SIZE, random_state=i)
    splits.append((pd.Index(tr_idx), pd.Index(te_idx)))

joblib.dump(splits, os.path.join(OUT_DIR, "splits_indices.joblib"))

# Sanity check
for i, (tr, te) in enumerate(splits[:5]):
    train_dyes = set(groups.loc[tr])
    test_dyes  = set(groups.loc[te])
    overlap = train_dyes & test_dyes
    assert len(overlap) == 0, f"Split {i}: {len(overlap)} dyes leaked!"
    print(f'Split {i}: Train={len(tr)} samples ({len(tr)/n_total:.1%}, {len(train_dyes)} dyes), '
          f'Test={len(te)} samples ({len(te)/n_total:.1%}, {len(test_dyes)} dyes)')
print("All splits are leak-free.\n")

# ──────────────────────────────────────────────
# Load best params from optimization
# ──────────────────────────────────────────────
params_path = os.path.join(OUT_DIR, "best_params_ks.joblib")
if not os.path.exists(params_path):
    print(f"ERROR: {params_path} not found. Run optimize.py first.")
    sys.exit(1)

best_params_all = joblib.load(params_path)
print(f"Loaded {len(best_params_all)} param sets from {params_path}")

# ──────────────────────────────────────────────
# Family map (build functions only)
# ──────────────────────────────────────────────
build_gnn = partial(build_final_pipeline_gnn,
                    smiles_series=smiles_df["SMILES"],
                    graph_cache=graph_cache)

family_map = {
    "mlp": (build_final_pipeline,    "mlp"),
    "rf":  (build_final_pipeline_rf, "rf"),
    "hgb": (build_final_pipeline_hgb, "hgb"),
    "gnn": (build_gnn,               "gnn"),
}

# ──────────────────────────────────────────────
# Train across all splits
# ──────────────────────────────────────────────
metrics_rows = []
predictions_dict = {}

for fam_name in MODEL_FAMILIES:
    if fam_name not in family_map:
        raise ValueError(f"Unknown family '{fam_name}'")

    build_fn, fam_tag = family_map[fam_name]

    if fam_tag == "gnn":
        X = dye_device_descs[device_cols]
    else:
        X = dye_device_descs

    for impute_method in IMPUTE_METHODS:
        key = f"{fam_tag}_{impute_method}"
        impute_label = impute_method if impute_method is not None else "KNN_imputation"

        if key not in best_params_all:
            print(f"WARNING: No params found for {key}, skipping.")
            continue

        best_params = best_params_all[key]
        print(f"Training {fam_tag} + {impute_label} across {N_SPLITS} splits...")

        for i, (tr_idx, te_idx) in enumerate(splits):
            X_train = X.loc[tr_idx]
            y_train = pce.loc[tr_idx]
            X_test  = X.loc[te_idx]
            y_test  = pce.loc[te_idx]

            pipe = build_fn(
                best_params, X_train,
                impute_strategy=impute_method,
                random_state=i
            )

            # GNN supports extra training args
            if fam_tag == "gnn":
                pipe.fit(X_train, y_train, max_epochs=500, patience=30)
            else:
                pipe.fit(X_train, y_train)

            pred_tr = pipe.predict(X_train)
            pred_te = pipe.predict(X_test)

            metrics_rows.append({
                "family": fam_tag,
                "impute_method": impute_label,
                "split": i,
                "R2_train": r2_score(y_train, pred_tr),
                "MAE_train": mean_absolute_error(y_train, pred_tr),
                "R2_test":  r2_score(y_test, pred_te),
                "MAE_test": mean_absolute_error(y_test, pred_te),
            })

            if (not SAVE_MODELS_ONLY_I0) or (i == 0):
                path = os.path.join(OUT_DIR, f"{fam_tag}_pipeline_{impute_label}_{i}.joblib")
                joblib.dump(pipe, path)

            col = f"pred_{fam_tag}_{impute_label}_{i:02d}"
            full_pred = np.empty(len(idx_all), dtype=np.float64)
            full_pred[np.isin(idx_all, tr_idx)] = pred_tr
            full_pred[np.isin(idx_all, te_idx)] = pred_te
            predictions_dict[col] = full_pred

            print(f"  Split {i}: R2_test={r2_score(y_test, pred_te):.4f}, "
                  f"MAE_test={mean_absolute_error(y_test, pred_te):.4f}")

# ──────────────────────────────────────────────
# Save results
# ──────────────────────────────────────────────
preds_df = pd.DataFrame(predictions_dict, index=idx_all)
preds_df.insert(0, "y_true", pce)

metrics_df = pd.DataFrame(metrics_rows)
metrics_df["model_id"] = (
    metrics_df["family"] + "_" + metrics_df["impute_method"]
    + "_split" + metrics_df["split"].astype(str)
)
metrics_df = metrics_df[
    ["model_id", "family", "impute_method", "split",
     "R2_train", "MAE_train", "R2_test", "MAE_test"]
]

metrics_df.to_csv(os.path.join(OUT_DIR, "metrics_summary.csv"), index=False)
preds_df.to_csv(os.path.join(OUT_DIR, "raw_predictions.csv"))

print(f"\nSaved metrics ({len(metrics_df)} rows) and predictions ({preds_df.shape})")

# ──────────────────────────────────────────────
# Print summary
# ──────────────────────────────────────────────
summary = (
    metrics_df
    .groupby(["family", "impute_method"])
    [["R2_train", "MAE_train", "R2_test", "MAE_test"]]
    .agg(["mean", "std"])
    .sort_values(("MAE_test", "mean"))
)
print("\n" + summary.to_string())
print("\nTraining complete.")