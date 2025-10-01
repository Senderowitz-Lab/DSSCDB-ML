import marimo

__generated_with = "0.14.12"
app = marimo.App(width="medium")


@app.cell
def _():

    import cheminfo_functions as cheminfo
    from joblib import Parallel, delayed
    import multiprocessing
    import pandas as pd
    import numpy as np
    from sklearn.metrics import r2_score,  root_mean_squared_error
    import os

    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.neural_network import MLPRegressor
    from sklearn.model_selection import KFold, cross_val_score, train_test_split
    from sklearn.metrics import make_scorer, r2_score, mean_absolute_error
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.ensemble import RandomForestRegressor
    import optuna
    from optuna.integration import KerasPruningCallback

    import joblib
    from joblib import parallel_backend
    from typing import Iterable


    # constants
    N_TRIALS = 100

    return (
        ColumnTransformer,
        Iterable,
        KFold,
        KNNImputer,
        MLPRegressor,
        RandomForestRegressor,
        StandardScaler,
        cross_val_score,
        joblib,
        make_pipeline,
        make_scorer,
        mean_absolute_error,
        np,
        optuna,
        os,
        pd,
        r2_score,
        root_mean_squared_error,
        train_test_split,
    )


@app.cell
def _(
    ColumnTransformer,
    KFold,
    KNNImputer,
    MLPRegressor,
    RandomForestRegressor,
    StandardScaler,
    cross_val_score,
    make_pipeline,
    make_scorer,
    np,
    optuna,
    pd,
    root_mean_squared_error,
):
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

        # Pipelines
        cont_pipe = make_pipeline(
            KNNImputer(n_neighbors=10),
            StandardScaler()
        )
        bin_pipe = make_pipeline(
            KNNImputer(n_neighbors=10) 
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

    def mlp_objective(trial, X, y, impute_strategy):
        # ----- hyper‑parameter search space -----
        n_layers = trial.suggest_int("n_layers", 1, 4)
        hidden = [
            trial.suggest_int(f"n_units_{i}", 10, 100) for i in range(n_layers)
        ]
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

        cv = KFold(n_splits=3, shuffle=True, random_state=trial.number)
        scores = cross_val_score(pipe, X, y, cv=cv, scoring=neg_rmse, n_jobs=1)

        # scores are NEGATIVE RMSE; minimise POSITIVE RMSE
        return -np.mean(scores)

    def rf_objective(trial, X, y, impute_strategy):
        # ----- search space -----
        n_estimators = trial.suggest_int("n_estimators", 50, 800, step=50)
        max_depth = trial.suggest_categorical("max_depth", [None, 6, 10, 16, 24, 32, 48])
        max_features = trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5, 0.7, 1.0])
        min_samples_split = trial.suggest_int("min_samples_split", 2, 20)
        min_samples_leaf  = trial.suggest_int("min_samples_leaf", 1, 20)
        bootstrap = trial.suggest_categorical("bootstrap", [True, False])
        max_samples = trial.suggest_float("max_samples", 0.5, 1.0) if bootstrap else None

        # ----- pipeline (preprocessor fit inside each CV fold; no leakage) -----
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

        cv = KFold(n_splits=3, shuffle=True, random_state=trial.number)
        scores = cross_val_score(
            pipe, X, y,
            cv=cv,
            scoring=neg_rmse, 
            n_jobs=1  # avoid nested parallelism (RF already uses n_jobs=-1)
        )

        return -np.mean(scores)

    def optuna_mlp_study(X, y, n_trials=100, n_jobs=4, db_path=None, impute_strategy='median'):
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
            lambda trial: mlp_objective(trial, X, y, impute_strategy),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )

        print("Best params:", study.best_params)
        return study.best_params

    def optuna_rf_study(X, y,n_trials=100,n_jobs = 4, db_path = None, impute_strategy= "median"):
        study = optuna.create_study(
            direction="minimize",
            study_name=f"Optuna-RandomForest-{impute_strategy}",
            storage=f"sqlite:///{db_path}" if db_path else None,
            load_if_exists=bool(db_path),
            pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=50),
        )

        study.optimize(
            lambda trial: rf_objective(trial, X, y, impute_strategy),
            n_trials=n_trials,
            n_jobs=n_jobs,
        )
        print("Best params:", study.best_params)
        return study.best_params
    return build_preprocessor, optuna_mlp_study, optuna_rf_study


@app.cell
def _(RandomForestRegressor, build_preprocessor, make_pipeline):

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

        # 3) Estimator + final pipeline
        model = RandomForestRegressor(**rf_kwargs)
        pipe = make_pipeline(preprocessor, model)
        return pipe

    return (build_final_pipeline_rf,)


@app.cell
def _(MLPRegressor, build_preprocessor, make_pipeline):
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
    return (build_final_pipeline,)


@app.cell
def _(pd):
    smiles_df = pd.read_csv("Data/cleaned_SMILES.csv", index_col=0)
    pce = smiles_df["PCE"]
    dye_device_descs = pd.read_csv("Data/dye_device_fp_not_imputed.csv", index_col=0)
    return dye_device_descs, pce


@app.cell
def _(
    Iterable,
    build_final_pipeline,
    build_final_pipeline_rf,
    joblib,
    mean_absolute_error,
    optuna_mlp_study,
    optuna_rf_study,
    os,
    pd,
    r2_score,
    train_test_split,
):

    def main_workflow_collect_predictions(
        unimputed_descriptors: pd.DataFrame,
        target: pd.Series,
        *,
        impute_methods: Iterable[str] = ("mean", "median", "most_frequent"),
        model_families: Iterable[str] = ("mlp",),      # use ("rf",) or ("mlp","rf")
        n_trials: int = 5,
        n_jobs_optuna: int = 10,
        save_models_only_i0: bool = True,
        out_dir: str = "models",
    ):
        """
        Returns:
          metrics_df : columns [model_id, family, impute_method, split, R2_train, MAE_train, R2_test, MAE_test]
          preds_df   : columns ['y_true', 'pred_<family>_<impute>_<split>'] for all molecules (index preserved)
        """
        os.makedirs(out_dir, exist_ok=True)

        # Map family -> (study_fn, build_fn, tag)
        family_map = {
            "mlp": (optuna_mlp_study,      build_final_pipeline,    "mlp"),
            "rf":  (optuna_rf_study,       build_final_pipeline_rf, "rf"),
        }
        families = []
        for fam in model_families:
            if fam not in family_map:
                raise ValueError(f"Unknown model family '{fam}'. Use 'mlp' or 'rf'.")
            families.append((fam, *family_map[fam]))

        # 1) Build 100 reproducible splits ON INDICES (shared by all models & imputations)
        idx_all = unimputed_descriptors.index
        splits = []
        for i in range(100):
            # standard random split on index labels
            tr_idx, te_idx = train_test_split(
                idx_all, test_size=0.2, random_state=i
                )
            tr_idx, te_idx = pd.Index(tr_idx), pd.Index(te_idx)
            splits.append((tr_idx, te_idx))

        joblib.dump(splits, "models/splits_indices.joblib")

        # 2) Containers
        metrics_rows = []
        preds_df = pd.DataFrame(index=idx_all)
        preds_df["y_true"] = target

        # Cache best params per (family, impute_method)
        best_cache = {}  # key: (family, impute_method) -> dict of best_params

        # 3) Loop families × imputations × splits
        for fam_name, study_fn, build_fn, fam_tag in families:
            for impute_method in impute_methods:

                # Optimize once per (family, impute_method) on split 0
                key = (fam_name, impute_method)
                if key not in best_cache:
                    tr0, te0 = splits[0]
                    X0 = unimputed_descriptors.loc[tr0]
                    y0 = target.loc[tr0]
                    best_obj = study_fn(
                        X0, y0,
                        n_trials=n_trials,
                        n_jobs=n_jobs_optuna,
                        impute_strategy=None, ##Impute method
                    )
                    best_params = getattr(best_obj, "best_params", best_obj)  # accept Study or dict
                    best_cache[key] = best_params

                best_params = best_cache[key]

                # Train / predict for all 100 splits
                for i, (tr_idx, te_idx) in enumerate(splits):
                    X_train = unimputed_descriptors.loc[tr_idx]
                    y_train = target.loc[tr_idx]
                    X_test  = unimputed_descriptors.loc[te_idx]
                    y_test  = target.loc[te_idx]

                    pipe = build_fn(
                        best_params, X_train,
                        impute_strategy=None, ##Impute method
                        random_state=i
                    )
                    pipe.fit(X_train, y_train)

                    # Metrics
                    pred_tr = pipe.predict(X_train)
                    pred_te = pipe.predict(X_test)
                    metrics_rows.append({
                        "family": fam_tag,
                        "impute_method": impute_method,
                        "split": i,
                        "R2_train": r2_score(y_train, pred_tr),
                        "MAE_train": mean_absolute_error(y_train, pred_tr),
                        "R2_test":  r2_score(y_test,  pred_te),
                        "MAE_test": mean_absolute_error(y_test,  pred_te),
                    })

                    # Save model(s)
                    if (not save_models_only_i0) or (i == 0):
                        path = os.path.join(out_dir, f"{fam_tag}_pipeline_{impute_method}_{i}.joblib")
                        joblib.dump(pipe, path)

                    # Raw predictions for ALL molecules with this model
                    col = f"pred_{fam_tag}_{impute_method}_{i:02d}"
                    preds_df[col] = pipe.predict(unimputed_descriptors)

        # 4) Finalize metrics_df
        metrics_df = pd.DataFrame(metrics_rows)
        metrics_df["model_id"] = (
            metrics_df["family"] + "_" + metrics_df["impute_method"] + "_split" + metrics_df["split"].astype(str)
        )
        metrics_df = metrics_df[
            ["model_id","family","impute_method","split","R2_train","MAE_train","R2_test","MAE_test"]
        ]

        # 5) (Optional) persist
        metrics_df.to_csv(os.path.join(out_dir, "metrics_summary.csv"), index=False)
        preds_df.to_csv(os.path.join(out_dir, "raw_predictions.csv"))

        return metrics_df, preds_df
    return (main_workflow_collect_predictions,)


@app.cell
def _(dye_device_descs, main_workflow_collect_predictions, pce):
    metrics_df, preds_df = main_workflow_collect_predictions(
        dye_device_descs, pce,
        model_families=('rf',),
        impute_methods=("KNN",),
        n_trials=100, n_jobs_optuna=10, save_models_only_i0=True
    )
    return


if __name__ == "__main__":
    app.run()
