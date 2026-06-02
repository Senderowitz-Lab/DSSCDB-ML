"""
SHAP.py — SHAP analysis for the best-performing model on the Kennard-Stone split.

Expects:
    - Output/best_params_ks.joblib
    - Data/cleaned_SMILES_fixed.csv
    - Data/dye_device_pce_without_dupes_new.csv

Produces:
    - Output/SHAP/shap_values.joblib
    - Output/SHAP/bar_global.png
    - Output/SHAP/beeswarm_overall.png
    - Output/SHAP/beeswarm_low.png
    - Output/SHAP/beeswarm_medium.png
    - Output/SHAP/beeswarm_high.png
    - Output/SHAP/dependence_top6.png
    - Output/SHAP/feature_importance_shap.csv
"""

import os
import sys
import numpy as np
import pandas as pd
import joblib
import shap
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.metrics.pairwise import euclidean_distances

from matplotlib.gridspec import GridSpec

from rdkit import Chem
from rdkit.Chem import Descriptors

from optuna_objectives import build_final_pipeline_hgb, build_preprocessor
from model_explain_functions import (
    retrieveSHAPBits_MolBI, resolveSHAPBits, shapBit_image,
    depictBit_new
)

# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
BEST_FAMILY = "hgb"
BEST_IMPUTE = "mean"
OUT_DIR = "Output/SHAP"
TEST_SIZE = 0.2
MAX_DISPLAY = 10

os.makedirs(OUT_DIR, exist_ok=True)


# ──────────────────────────────────────────────
# Helper: annotate a beeswarm axis with
#         substructure images
# ──────────────────────────────────────────────

def beeswarm_with_substructures(explanation, X_test, smiles_df,
                                 max_display=10, title="",
                                 figsize=(35.0, 22.5), save_path=None):
    """
    Three-column layout:
      Left column:   consensus substructure fragments
      Center column: SHAP beeswarm plot
      Right column:  example molecules with highlighted bits
    """

    # ── Layout: [fragments | beeswarm | molecules] ──
    fig = plt.figure(figsize=figsize)
    gs = GridSpec(1, 3, width_ratios=[1, 3, 2.5], wspace=0.05, figure=fig)

    ax_left = fig.add_subplot(gs[0])
    ax_center = fig.add_subplot(gs[1])
    ax_right = fig.add_subplot(gs[2])

    # Turn off axes for image columns
    ax_left.axis('off')
    ax_right.axis('off')

    # ── Center: beeswarm ──
    shap.plots.beeswarm(explanation, max_display=max_display, show=False,
                         color_bar=False, group_remaining_features=False,
                         ax=ax_center, plot_size=None)

    ylabels = ax_center.get_yticklabels()
    ylabels_text = [label.get_text() for label in ylabels]

    # Get y positions in data coords (these are just 0, 1, 2, ... for each feature)
    y_positions = {label.get_text(): label.get_position()[1] for label in ylabels}

    # ── Resolve substructures ──
    shap_bit_mol_bi = retrieveSHAPBits_MolBI(ylabels_text, X_test, smiles_df)
    if not shap_bit_mol_bi:
        if title:
            fig.suptitle(title, fontsize=12, fontweight="bold")
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        return

    shapBit_smarts, shapBit_smarts_maj, shapBit_smiles_maj = resolveSHAPBits(shap_bit_mol_bi)
    shapBit_image_dict = shapBit_image(shapBit_smiles_maj, molSize=(400, 180))

    # ── Sync y-axis limits across all three columns ──
    ymin, ymax = ax_center.get_ylim()
    ax_left.set_ylim(ymin, ymax)
    ax_right.set_ylim(ymin, ymax)

    # ── Left column: consensus fragments ──
    for shapBit in shapBit_image_dict:
        shapBit_label = f"Bit_{shapBit}"
        if shapBit_label not in y_positions:
            continue
        y = y_positions[shapBit_label]
        im = plt.imread(shapBit_image_dict[shapBit])
        ib = OffsetImage(im, zoom=0.45)
        ib.image.axes = ax_left
        ab = AnnotationBbox(
            ib, (0.4, y),
            xycoords=('axes fraction', 'data'),
            frameon=False
        )
        ax_left.add_artist(ab)

    # ── Right column: example molecules ──
    used_mols = []
    mol_idx = 0
    for shapBit in shap_bit_mol_bi:
        if shapBit not in shapBit_smarts_maj:
            continue
        shapBit_label = f"Bit_{shapBit}"
        if shapBit_label not in y_positions:
            continue

        smarts = shapBit_smarts_maj[shapBit][0]
        molCandidates = []
        mws = []
        for smiles in shap_bit_mol_bi[shapBit]:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                continue
            smart_query = Chem.MolFromSmarts(smarts)
            if smart_query is None:
                continue
            if mol.HasSubstructMatch(smart_query):
                canonSmiles = Chem.MolToSmiles(mol)
                if canonSmiles in used_mols:
                    continue
                molCandidates.append(mol)
                mws.append(Descriptors.HeavyAtomMolWt(mol))

        if not molCandidates:
            continue

        lowMw_mol = molCandidates[pd.Series(mws).idxmin()]
        used_mols.append(Chem.MolToSmiles(lowMw_mol))

        y = y_positions[shapBit_label]
        im = plt.imread(depictBit_new(bitId=shapBit, mol=lowMw_mol, molSize=(600, 250)))
        ib = OffsetImage(im, zoom=0.75)
        ib.image.axes = ax_right

        # Stagger:
        x_pos = 0.2 if mol_idx % 2 == 0 else 0.7

        ab = AnnotationBbox(
            ib, (x_pos, y),
            xycoords=('axes fraction', 'data'),
            frameon=False
        )
        ax_right.add_artist(ab)
        mol_idx += 1

    # ── Styling ──
    ax_center.tick_params(axis='x', labelsize=22)
    ax_center.tick_params(axis='y', labelsize=22)
    ax_center.set_xlabel(ax_center.xaxis.get_label().get_text(), fontsize=22, labelpad=10)

    if title:
        fig.suptitle(title, fontsize=16)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.close(fig)

def style_beeswarm_axes(ax):
    """Apply consistent font sizes to a beeswarm axis."""
    ax.set_xticklabels(ax.xaxis.get_ticklabels(), fontsize=24)
    ax.set_yticklabels(ax.yaxis.get_ticklabels(), fontsize=24)
    ax.set_xlabel(ax.xaxis.get_label().get_text(), fontsize=26, labelpad=10)


# ──────────────────────────────────────────────
# Load data
# ──────────────────────────────────────────────
print("Loading data...")
smiles_df = pd.read_csv("Data/cleaned_SMILES_fixed.csv", index_col=0)
dye_device_descs = pd.read_csv("Data/dye_device_pce_without_dupes_new.csv", index_col=0)
pce = dye_device_descs['PCE'].dropna()
dye_device_descs = dye_device_descs.drop('PCE', axis=1).copy().loc[pce.index]
smiles_df = smiles_df.loc[dye_device_descs.index]

fp_cols = [c for c in dye_device_descs.columns if c.startswith("Bit_")]
device_cols = [c for c in dye_device_descs.columns if not c.startswith("Bit_")]
groups = smiles_df.loc[dye_device_descs.index, 'SMILES']

print(f"Dataset: {len(dye_device_descs)} samples, "
      f"{len(fp_cols)} fingerprint cols, {len(device_cols)} device cols")


# ──────────────────────────────────────────────
# Kennard-Stone split
# ──────────────────────────────────────────────
def kennard_stone_groups(X_df, groups, fp_cols, test_size=0.2):
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
    return train_idx, test_idx


print("Building Kennard-Stone split...")
train_idx, test_idx = kennard_stone_groups(
    dye_device_descs, groups, fp_cols, test_size=TEST_SIZE
)
n_total = len(dye_device_descs)
print(f"  Train: {len(train_idx)} samples ({len(train_idx)/n_total:.1%})")
print(f"  Test:  {len(test_idx)} samples ({len(test_idx)/n_total:.1%})")

X_train = dye_device_descs.loc[train_idx]
X_test = dye_device_descs.loc[test_idx]
y_train = pce.loc[train_idx]
y_test = pce.loc[test_idx]


# ──────────────────────────────────────────────
# Build and train the best model
# ──────────────────────────────────────────────
print(f"\nLoading best params for {BEST_FAMILY} + {BEST_IMPUTE}...")
params_path = os.path.join("Output", "best_params_ks.joblib")
best_params_all = joblib.load(params_path)
key = f"{BEST_FAMILY}_{BEST_IMPUTE}"
best_params = best_params_all[key]
print(f"  Params: {best_params}")

print("Training model...")
pipe = build_final_pipeline_hgb(best_params, X_train,
                                 impute_strategy=BEST_IMPUTE, random_state=42)
pipe.fit(X_train, y_train)

pred_train = pipe.predict(X_train)
pred_test = pipe.predict(X_test)
print(f"  Train R²={r2_score(y_train, pred_train):.4f}, "
      f"MAE={mean_absolute_error(y_train, pred_train):.4f}")
print(f"  Test  R²={r2_score(y_test, pred_test):.4f}, "
      f"MAE={mean_absolute_error(y_test, pred_test):.4f}")

joblib.dump(pipe, os.path.join(OUT_DIR, "best_model_ks.joblib"))


# ──────────────────────────────────────────────
# Extract preprocessor and estimator
# ──────────────────────────────────────────────
preprocessor = pipe[0]
estimator = pipe[-1]

X_train_t = preprocessor.transform(X_train)
X_test_t = preprocessor.transform(X_test)

try:
    feat_names = preprocessor.get_feature_names_out()
except Exception:
    feat_names = [f"feat_{i}" for i in range(X_train_t.shape[1])]

# Rename features for display
rename_map = {
    'Scatt': 'scattering_thickness_um',
}
feat_names = [rename_map.get(f, f) for f in feat_names]

X_train_t_df = pd.DataFrame(X_train_t, index=X_train.index, columns=feat_names)
X_test_t_df = pd.DataFrame(X_test_t, index=X_test.index, columns=feat_names)

print(f"\nTransformed features: {X_test_t_df.shape[1]}")


# ──────────────────────────────────────────────
# SHAP: TreeExplainer
# ──────────────────────────────────────────────
print("\nComputing SHAP values with TreeExplainer...")
explainer = shap.TreeExplainer(estimator)
shap_values = explainer.shap_values(X_test_t_df)

explanation = shap.Explanation(
    values=shap_values,
    base_values=explainer.expected_value,
    data=X_test_t_df.values,
    feature_names=list(feat_names),
)

joblib.dump({
    'explanation': explanation,
    'shap_values': shap_values,
    'base_values': explainer.expected_value,
    'feature_names': list(feat_names),
    'X_test_index': X_test.index,
    'pred_test': pred_test,
}, os.path.join(OUT_DIR, "shap_values.joblib"))
print(f"  Saved SHAP values to {OUT_DIR}/shap_values.joblib")


# ──────────────────────────────────────────────
# Plot 1: Global bar plot (no substructures)
# ──────────────────────────────────────────────
print("\nGenerating global bar plot...")
fig, ax = plt.subplots(figsize=(12, 8))
shap.plots.bar(explanation, max_display=MAX_DISPLAY, show=False, ax=ax)
ax.set_xlabel("mean(|SHAP value|)", fontsize=14)
ax.tick_params(labelsize=12)
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "bar_global.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved bar_global.png")


# ──────────────────────────────────────────────
# Plot 2: Beeswarm — overall (with substructures)
# ──────────────────────────────────────────────
print("Generating beeswarm (overall) with substructures...")
beeswarm_with_substructures(
    explanation, X_test, smiles_df,
    title=None, #f"SHAP — Overall Test PCE [{pred_test.min():.2f} to {pred_test.max():.2f}]",
    save_path=os.path.join(OUT_DIR, "beeswarm_overall.png")
)
print("  Saved beeswarm_overall.png")


# ──────────────────────────────────────────────
# Plot 3: Beeswarm by PCE range (with substructures)
# ──────────────────────────────────────────────
print("Generating beeswarm plots by PCE range with substructures...")
pred_series = pd.Series(pred_test, index=X_test.index)
range_labels = pd.cut(pred_series, 3, labels=["Low", "Medium", "High"])

# ── Per range ──
for label in ["Low", "Medium", "High"]:
    mask = range_labels == label
    if mask.sum() == 0:
        continue

    mask_arr = mask.values
    pred_subset = pred_series[mask]
    X_test_subset = X_test.loc[mask]

    expl_subset = shap.Explanation(
        values=shap_values[mask_arr],
        base_values=explainer.expected_value,
        data=X_test_t_df.values[mask_arr],
        feature_names=list(feat_names),
    )

    beeswarm_with_substructures(
        expl_subset, X_test_subset, smiles_df,
        title=None, #f"SHAP — Predicted PCE ({label}): [{pred_subset.min():.2f} to {pred_subset.max():.2f}]",
        save_path=os.path.join(OUT_DIR, f"beeswarm_{label.lower()}.png")
    )


# ──────────────────────────────────────────────
# Plot 4: Dependence plots for top 6 features
# ──────────────────────────────────────────────
print("Generating dependence plots for top features...")
mean_abs_shap = np.abs(shap_values).mean(axis=0)
top_indices = np.argsort(mean_abs_shap)[::-1][:6]
top_features = [feat_names[i] for i in top_indices]

fig, axes = plt.subplots(2, 3, figsize=(20, 12))
for idx, (feat, ax) in enumerate(zip(top_features, axes.ravel())):
    feat_idx = list(feat_names).index(feat)
    shap.plots.scatter(explanation[:, feat_idx], color=explanation, show=False, ax=ax)
    ax.set_title(feat, fontsize=13)
    ax.tick_params(labelsize=10)

plt.suptitle("SHAP Dependence — Top 6 Features", fontsize=16, fontweight='bold')
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, "dependence_top6.png"), dpi=200, bbox_inches='tight')
plt.close(fig)
print("  Saved dependence_top6.png")


# ──────────────────────────────────────────────
# Plot 5: Per dye-family SHAP beeswarms
# ──────────────────────────────────────────────
print("\nClassifying test set dyes by scaffold family...")

from rdkit.Chem.Scaffolds import MurckoScaffold

def _smarts(s):
    m = Chem.MolFromSmarts(s)
    if m is None:
        raise ValueError(f"Invalid SMARTS: {s}")
    return m

def _count_pyrrolic_like_N(mol):
    ri = mol.GetRingInfo()
    count = 0
    for ring in ri.AtomRings():
        if len(ring) != 5:
            continue
        if not all(mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            continue
        if any(mol.GetAtomWithIdx(i).GetAtomicNum() == 7
               and mol.GetAtomWithIdx(i).GetIsAromatic() for i in ring):
            count += 1
    return count

PATTERNS = {
    "ruthenium": [_smarts("[Ru]")],
    "squaraine": [
        _smarts("O=C1C(=O)C(=O)C(=O)1"),
        _smarts("O=C1C(=O)C(O)=C(=O)1"),
        _smarts("O=C1C(=O)C([O-])=C(=O)1"),
        _smarts("O=C1C(=O)C(=C*)C(=O)1"),
        _smarts("O=C1C(=*)C(=O)C(=O)1"),
        _smarts("C1(=O)C(=O)C(=O)C(=O)1"),
    ],
    "phenothiazine": [
        _smarts("c1ccc2nc3ccccc3sc2c1"),
        _smarts("c1ccc2sc3ccccn3c2c1"),
        _smarts("[nH]1c2ccccc2sc3ccccc13"),
        _smarts("c1ccc2nc3ccccc3s(=O)c2c1"),
        _smarts("c1ccc2nc3ccccc3s(=O)(=O)c2c1"),
        _smarts("c1ccc2c(c1)Nc1ccccc1S2"),
    ],
    "carbazole": [
        _smarts("c1ccc2c(c1)[nH]c3ccccc23"),
        _smarts("n1c2ccccc2c3ccccc13"),
        _smarts("[nH]1c2ccccc2c3ccccc13"),
    ],
    "triphenylamine": [
        _smarts("[N;X3;H0;!$(*=O);!$([N+]);!R]([c;a])([c;a])[c;a]"),
    ],
    "indoline": [
        _smarts("[NH]1Cc2ccccc2C1"),
        _smarts("[N;X3]1Cc2ccccc2CC1"),
        _smarts("c1ccc2NCCCc2c1"),
        _smarts("c1cccc2NCCCc12"),
        _smarts("c1cc2CCCCN2c1"),
    ],
    "coumarin": [
        _smarts("O=c1occc2ccccc12"),
        _smarts("O=C1Oc2ccccc2C=C1"),
        _smarts("O=C1OC=CC2=CC=CC=C12"),
        _smarts("O=C1C=CC(=O)Oc2ccccc12"),
    ],
}

FAMILY_PRIORITY = [
    "ruthenium", "porphyrin", "squaraine", "phenothiazine",
    "carbazole", "indoline", "coumarin", "triphenylamine",
]

PRETTY_NAME = {
    "ruthenium": "ruthenium", "porphyrin": "porphyrin",
    "squaraine": "squaraine", "phenothiazine": "phenothiazine",
    "carbazole": "carbazole", "triphenylamine": "triphenylamine",
    "indoline": "indoline", "coumarin": "coumarin", "misc": "miscellaneous",
}


def classify_smiles(smi):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "miscellaneous"
    hits = set()
    try:
        if _count_pyrrolic_like_N(mol) >= 4:
            hits.add("porphyrin")
    except Exception:
        pass
    for fam, patt_list in PATTERNS.items():
        for patt in patt_list:
            if mol.HasSubstructMatch(patt):
                hits.add(fam)
                break
    if not hits:
        return "miscellaneous"
    for fam in FAMILY_PRIORITY:
        if fam in hits:
            return PRETTY_NAME.get(fam, fam)
    return "miscellaneous"


# Tag test set SMILES
smiles_test = smiles_df['SMILES'].loc[X_test.index]
family_labels = smiles_test.apply(classify_smiles)

print("  Family distribution in test set:")
print(family_labels.value_counts().to_string())

# Generate per-family beeswarm plots
FAMILIES_TO_PLOT = ["triphenylamine", "carbazole", "phenothiazine"]

for family in FAMILIES_TO_PLOT:
    mask = family_labels.eq(family)
    n_samples = mask.sum()

    if n_samples < 5:
        print(f"  Skipping {family} — only {n_samples} samples")
        continue

    mask_arr = mask.reindex(X_test.index).fillna(False).to_numpy()

    X_test_fam = X_test.loc[mask]
    pred_fam = pred_series[mask]

    expl_fam = shap.Explanation(
        values=shap_values[mask_arr],
        base_values=explainer.expected_value,
        data=X_test_t_df.values[mask_arr],
        feature_names=list(feat_names),
    )

    fname = f"beeswarm_{family.lower()}.png"
    beeswarm_with_substructures(
        expl_fam, X_test_fam, smiles_df,
        title=None, #f"SHAP — {family.capitalize()} (n={n_samples})",
        save_path=os.path.join(OUT_DIR, fname)
    )
    print(f"  Saved {fname} ({n_samples} samples)")


# ──────────────────────────────────────────────
# Summary statistics
# ──────────────────────────────────────────────
print("\n" + "="*60)
print(f"TOP {MAX_DISPLAY} FEATURES BY MEAN |SHAP|")
print("="*60)
importance_df = pd.DataFrame({
    'feature': feat_names,
    'mean_abs_shap': np.abs(shap_values).mean(axis=0),
    'std_abs_shap': np.abs(shap_values).std(axis=0),
}).sort_values('mean_abs_shap', ascending=False)

importance_df['type'] = importance_df['feature'].apply(
    lambda f: 'fingerprint' if f.startswith('Bit_') else 'device'
)

print(importance_df.head(MAX_DISPLAY).to_string(index=False))

type_summary = importance_df.groupby('type')['mean_abs_shap'].agg(['sum', 'mean', 'count'])
print(f"\n{'='*60}")
print("SHAP CONTRIBUTION BY FEATURE TYPE")
print("="*60)
print(type_summary.to_string())

importance_df.to_csv(os.path.join(OUT_DIR, "feature_importance_shap.csv"), index=False)
print(f"\nSaved feature_importance_shap.csv")

print("\nSHAP analysis complete.")