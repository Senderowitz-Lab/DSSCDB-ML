import pandas as pd
import numpy as np
import optuna

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.model_selection import KFold, cross_val_score, GroupKFold, GroupShuffleSplit
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import make_scorer, r2_score, mean_absolute_error, root_mean_squared_error


# Negative RMSE scorer so cross_val_score "higher is better"
neg_rmse = make_scorer(root_mean_squared_error, greater_is_better=False)

def _split_binary_continuous(X):
    """
    Return (binary_cols, continuous_cols) by inspecting a pandas DataFrame.
    A column is considered binary if its non-null unique values are subset of {0,1}.
    Booleans count as binary too.
    """
    if not isinstance(X, pd.DataFrame):
        # If not a DataFrame, we can't inspect columns—treat all as continuous
        n_features = X.shape[1]
        return [], list(range(n_features))

    binary_cols = []
    continuous_cols = []
    for col in X.columns:
        s = X[col]
        # Drop NaNs, get unique values
        vals = pd.unique(s.dropna())
        # Coerce booleans to ints for the check
        vals_coerced = pd.Series(vals).astype('int64', errors='ignore')
        try:
            # Set of {0,1}?
            is_binary = set(pd.Series(vals_coerced).dropna().astype(int)).issubset({0, 1})
        except Exception:
            is_binary = False

        if is_binary:
            binary_cols.append(col)
        else:
            continuous_cols.append(col)
    return binary_cols, continuous_cols

def build_preprocessor(X, impute_strategy=None):
    bin_cols, cont_cols = _split_binary_continuous(X)

    if impute_strategy is None:
        impute_method = KNNImputer(n_neighbors=10)
    else:
        impute_method = SimpleImputer(strategy=impute_strategy)

    # Pipelines
    cont_pipe = make_pipeline(
        impute_method,
        StandardScaler()
    )
    bin_pipe = make_pipeline(
        impute_method
        # no scaler here
    )
    if isinstance(X, pd.DataFrame):
        pre = ColumnTransformer(
            transformers=[
                ("cont", cont_pipe, cont_cols),
                ("bin",  bin_pipe,  bin_cols),
            ],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )
    else:
        # X is ndarray -> all features continuous
        pre = ColumnTransformer(
            transformers=[("cont", cont_pipe, list(range(X.shape[1])))],
            remainder="drop",
            sparse_threshold=0.0,
            verbose_feature_names_out=False,
        )

    return pre

def mlp_objective(trial, X, y, impute_strategy, groups):
    # ----- hyper‑parameter search space -----
    n_layers = trial.suggest_int("n_layers", 1, 4)
    hidden = [trial.suggest_int(f"n_units_{i}", 10, 100) for i in range(n_layers)]
    alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
    lr0 = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
    act = trial.suggest_categorical("activation", ["relu", "tanh"])

    preprocessor = build_preprocessor(X, impute_strategy)

    pipe = make_pipeline(
        preprocessor,
        MLPRegressor(
            hidden_layer_sizes=tuple(hidden),
            activation=act,
            solver="adam",
            learning_rate_init=lr0,
            alpha=alpha,
            max_iter=800,
            early_stopping=True,
            n_iter_no_change=20,
            random_state=trial.number,
        ),
    )

    cv = GroupKFold(n_splits=3)
    scores = cross_val_score(pipe, X, y, cv=cv, groups=groups, scoring=neg_rmse, n_jobs=1)

    # scores are NEGATIVE RMSE; minimise POSITIVE RMSE
    return -np.mean(scores)

def rf_objective(trial, X, y, impute_strategy, groups):
    # ----- search space -----
    n_estimators = trial.suggest_int("n_estimators", 50, 800, step=50)
    max_depth = trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24, 32, 48])
    max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0])
    min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20)
    bootstrap = trial.suggest_categorical("bootstrap", [True, False])
    max_samples = trial.suggest_float("max_samples", 0.5, 1.0) if bootstrap else None

    # ----- pipeline  -----
    preprocessor = build_preprocessor(X, impute_strategy)

    rf = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        max_samples=max_samples,           # ignored if bootstrap=False
        n_jobs=-1,
        random_state=trial.number,
    )

    pipe = make_pipeline(preprocessor, rf)

    cv = GroupKFold(n_splits=3)
    scores = cross_val_score(
        pipe, X, y,
        cv=cv,
        groups=groups,
        scoring=neg_rmse, 
        n_jobs=1  # avoid nested parallelism (RF already uses n_jobs=-1)
    )

    return -np.mean(scores)

def histgb_objective(trial, X, y, impute_strategy, groups):
    max_iter          = trial.suggest_int("max_iter", 50, 800, step=50)
    max_depth         = trial.suggest_int("max_depth", 3, 16)
    learning_rate     = trial.suggest_float("learning_rate", 0.01, 0.3, log=True)
    max_leaf_nodes    = trial.suggest_int("max_leaf_nodes", 15, 127)
    min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 40)
    l2_regularization = trial.suggest_float("l2_regularization", 1e-6, 10.0, log=True)
    max_features      = trial.suggest_float("max_features", 0.3, 1.0)

    preprocessor = build_preprocessor(X, impute_strategy)

    hgb = HistGradientBoostingRegressor(
        max_iter=max_iter,
        max_depth=max_depth,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
        l2_regularization=l2_regularization,
        max_features=max_features,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=trial.number,
    )

    pipe = make_pipeline(preprocessor, hgb)

    cv = GroupKFold(n_splits=3)
    scores = cross_val_score(
        pipe, X, y,
        cv=cv, groups=groups,
        scoring=neg_rmse, n_jobs=1
    )
    return -np.mean(scores)

def optuna_mlp_study(X, y, groups, n_trials=100, n_jobs=4, db_path=None, impute_strategy='median'):
    study = optuna.create_study(
        direction="minimize",
        study_name=f"Optuna-MLPRegressor-{impute_strategy}",
        storage=f"sqlite:///{db_path}" if db_path else None,
        load_if_exists=bool(db_path),
        pruner=optuna.pruners.MedianPruner(
            n_startup_trials=10, n_warmup_steps=50
        ),
    )

    study.optimize(
        lambda trial: mlp_objective(trial, X, y, impute_strategy, groups),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )

    print("Best params:", study.best_params)
    return study.best_params

def optuna_rf_study(X, y, groups, n_trials=100,n_jobs = 4, db_path = None, impute_strategy= "median"):
    study = optuna.create_study(
        direction="minimize",
        study_name=f"Optuna-RandomForest-{impute_strategy}",
        storage=f"sqlite:///{db_path}" if db_path else None,
        load_if_exists=bool(db_path),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
    )

    study.optimize(
        lambda trial: rf_objective(trial, X, y, impute_strategy, groups),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    print("Best params:", study.best_params)
    return study.best_params

def optuna_hgb_study(X, y, groups, n_trials=100, n_jobs=4,
                    db_path=None, impute_strategy="median"):
    study = optuna.create_study(
        direction="minimize",
        study_name=f"Optuna-GradientBoosting-{impute_strategy}",
        storage=f"sqlite:///{db_path}" if db_path else None,
        load_if_exists=bool(db_path),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
    )
    study.optimize(
        lambda trial: histgb_objective(trial, X, y, impute_strategy, groups),
        n_trials=n_trials,
        n_jobs=n_jobs,
    )
    print("Best params:", study.best_params)
    return study.best_params

def build_final_pipeline_rf(best_params, X, impute_strategy, random_state=0):

    preprocessor = build_preprocessor(X, impute_strategy)

    # collect hyperparameters (with sensible fallbacks)
    n_estimators = int(best_params.get("n_estimators", 100))
    max_depth = best_params.get("max_depth", None)          # can be None or int
    max_features = best_params.get("max_features", "sqrt")  # "sqrt", "log2", or float in (0,1]
    min_samples_split = int(best_params.get("min_samples_split", 2))
    min_samples_leaf  = int(best_params.get("min_samples_leaf", 1))
    bootstrap = bool(best_params.get("bootstrap", True))

    rf_kwargs = dict(
        n_estimators=n_estimators,
        max_depth=max_depth,
        max_features=max_features,
        min_samples_split=min_samples_split,
        min_samples_leaf=min_samples_leaf,
        bootstrap=bootstrap,
        n_jobs=-1,
        random_state=random_state,
    )

    # Only set max_samples when bootstrapping
    max_samples = best_params.get("max_samples", None)
    if bootstrap and (max_samples is not None):
        rf_kwargs["max_samples"] = float(max_samples)

    # Estimator + final pipeline
    model = RandomForestRegressor(**rf_kwargs)
    pipe = make_pipeline(preprocessor, model)
    return pipe

def build_final_pipeline(best_params, X, impute_strategy, random_state=0):
    # Build hidden_layer_sizes tuple in order 0..n_layers-1
    n_layers = best_params["n_layers"]
    hidden = tuple(best_params[f"n_units_{i}"] for i in range(n_layers))

    preprocessor = build_preprocessor(X, impute_strategy)  

    model = MLPRegressor(
        hidden_layer_sizes=hidden,
        activation=best_params["activation"],
        solver="adam",
        learning_rate_init=best_params["learning_rate_init"],
        alpha=best_params["alpha"],
        max_iter=800,
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )

    pipe = make_pipeline(preprocessor, model)
    return pipe

def build_final_pipeline_hgb(best_params, X, impute_strategy, random_state=0):
    preprocessor = build_preprocessor(X, impute_strategy)

    hgb = HistGradientBoostingRegressor(
        max_iter=int(best_params.get("max_iter", 100)),
        max_depth=int(best_params.get("max_depth", 6)),
        learning_rate=float(best_params.get("learning_rate", 0.1)),
        max_leaf_nodes=int(best_params.get("max_leaf_nodes", 31)),
        min_samples_leaf=int(best_params.get("min_samples_leaf", 20)),
        l2_regularization=float(best_params.get("l2_regularization", 0.0)),
        max_features=float(best_params.get("max_features", 1.0)),
        early_stopping=True,
        n_iter_no_change=20,
        random_state=random_state,
    )

    return make_pipeline(preprocessor, hgb)

# ---- GNN ----
from gnn_model import GNNPipeline, precompute_graphs, ATOM_FEAT_DIM

def gnn_objective(trial, X, y, impute_strategy, groups,
                  smiles_series, graph_cache):
    hparams = {
        'gnn_hidden':    trial.suggest_int("gnn_hidden", 64, 128, step=32),
        'gnn_out':       trial.suggest_int("gnn_out", 32, 64, step=16),
        'n_gnn_layers':  trial.suggest_int("n_gnn_layers", 2, 4),
        'device_hidden': trial.suggest_int("device_hidden", 32, 64, step=16),
        'pred_hidden':   trial.suggest_int("pred_hidden", 32, 64, step=16),
        'dropout':       trial.suggest_float("dropout", 0.05, 0.3),
        'lr':            trial.suggest_float("lr", 5e-4, 3e-3, log=True),
    }

    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    tr_pos, te_pos = next(gss.split(X, y, groups))

    X_tr, X_te = X.iloc[tr_pos], X.iloc[te_pos]
    y_tr, y_te = y.iloc[tr_pos], y.iloc[te_pos]

    pipe = GNNPipeline(smiles_series, graph_cache, hparams,
                       impute_strategy=impute_strategy,
                       random_state=trial.number) 
    pipe.fit(X_tr, y_tr, max_epochs=200, patience=20)
    pred = pipe.predict(X_te)

    print("Pred stats:",
      "nan:", np.isnan(pred).any(),
      "inf:", np.isinf(pred).any(),
      "min:", np.nanmin(pred),
      "max:", np.nanmax(pred))

    if np.any(np.isnan(pred)) or np.any(np.isinf(pred)):
        return float('inf')

    return root_mean_squared_error(y_te.values, pred)


def optuna_gnn_study(X, y, groups, n_trials=30, n_jobs=1,
                     db_path=None, impute_strategy='median',
                     smiles_series=None, graph_cache=None):
    study = optuna.create_study(
        direction="minimize",
        study_name=f"Optuna-GNN-{impute_strategy}",
        pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
    )
    study.optimize(
        lambda trial: gnn_objective(
            trial, X, y, impute_strategy, groups,
            smiles_series, graph_cache
        ),
        n_trials=n_trials,
        n_jobs=n_jobs,   # keep at 1 unless you have multiple GPUs
    )
    print("Best params:", study.best_params)
    return study.best_params

def build_final_pipeline_gnn(best_params, X, impute_strategy,
                              random_state=0, smiles_series=None,
                              graph_cache=None):
    hparams = {k: best_params[k] for k in [
        'gnn_hidden', 'gnn_out', 'n_gnn_layers',
        'device_hidden', 'pred_hidden', 'dropout', 'lr'
    ]}
    return GNNPipeline(smiles_series, graph_cache, hparams, impute_strategy=impute_strategy,
                       random_state=random_state)