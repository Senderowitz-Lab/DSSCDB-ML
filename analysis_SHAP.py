import marimo

__generated_with = "0.15.2"
app = marimo.App(width="medium")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import numpy as np

    import re
    from io import BytesIO 

    from collections import Counter

    import shap

    from sklearn.metrics import r2_score, root_mean_squared_error

    import qsar_functions as qsar
    import model_explain_functions as explain

    import seaborn as sns

    import joblib
    from sklearn.model_selection import train_test_split

    from importlib import reload
    return explain, joblib, pd, qsar, reload


@app.cell
def _(explain, reload):
    reload(explain)
    return


@app.cell
def _(joblib, pd):
    dye_device_descs = pd.read_csv('Data/dye_device_fp_not_imputed.csv', index_col=0)
    smiles_df = pd.read_csv('Data/cleaned_SMILES.csv', index_col=0)
    pce = smiles_df['PCE']
    splits = joblib.load("models/splits_indices.joblib")  # list of (pd.Index, pd.Index)
    model_dir = 'models/rf_pipeline_KNN_0.joblib'
    return dye_device_descs, model_dir, pce, smiles_df, splits


@app.cell
def _(dye_device_descs, joblib, model_dir, pce, qsar, splits):
    pipe = joblib.load(model_dir)

    # Split
    X_train = dye_device_descs.loc[splits[0][0]]
    X_test = dye_device_descs.loc[splits[0][1]]
    y_train = pce.loc[splits[0][0]]
    y_test = pce.loc[splits[0][1]]

    #Visualization Split 0
    qsar_df, train_raw, test_raw = qsar.reg_results_df(pipe, X_train, X_test, y_train, y_test)
    qsar.reg_results_viz_sns(qsar_df, train_raw, test_raw, box_adjustment=0.15)
    return X_test, qsar_df


@app.cell
def _(dye_device_descs, explain, model_dir, pce, splits):
    #Get Shap Values for split 0
    explanation, X_eval, P_eval = explain.get_Explanation_object(model_dir, dye_device_descs, pce, splits[0], eval_size=1000)
    return P_eval, X_eval, explanation


@app.cell
def _(P_eval, X_eval, explain, explanation, smiles_df):
    explain.shap_mol_fp(explanation=explanation,
                X_test=X_eval, 
                P_test=P_eval, 
                title="Predicted PCE Range (Overall)", 
                smiles_df=smiles_df,
                plot_type='dot', 
                max_display=10
    )
    return


@app.cell
def _(pd, qsar_df):
    P_test = qsar_df.loc[qsar_df["Dataset"] == "Test"]["Predicted"]
    P_test_range_labels = pd.cut(P_test, 3, labels=["Low", "Medium", "High"])
    return P_test, P_test_range_labels


@app.cell
def _(P_test, P_test_range_labels, X_test, explanation):
    low_mask = P_test_range_labels.reindex(X_test.index).eq("Low").to_numpy()
    X_test_low = X_test.loc[P_test_range_labels == "Low"]
    P_test_low = P_test.loc[P_test_range_labels == "Low"]

    #slice the Explanation

    expl_low = explanation[low_mask]          # works in recent SHAP (row slicing)
    return P_test_low, X_test_low, expl_low


@app.cell
def _(P_test_low, X_test_low, expl_low, explain, smiles_df):
    explain.shap_mol_fp(explanation=expl_low,
                X_test=X_test_low, 
                P_test=P_test_low, 
                title="Predicted PCE Range (Low)", 
                smiles_df=smiles_df,
                plot_type='dot', 
                max_display=10
    )
    return


@app.cell
def _(P_test, P_test_range_labels, X_test, explain, explanation, smiles_df):
    med_mask = P_test_range_labels.reindex(X_test.index).eq("Medium").to_numpy()
    X_test_med = X_test.loc[P_test_range_labels == "Medium"]
    P_test_med = P_test.loc[P_test_range_labels == "Medium"]
    #slice the Explanation

    expl_med = explanation[med_mask]          # works in recent SHAP (row slicing)


    explain.shap_mol_fp(explanation=expl_med,
                X_test=X_test_med, 
                P_test=P_test_med, 
                title="Predicted PCE Range (Medium)", 
                smiles_df=smiles_df,
                plot_type='dot', 
                max_display=10
    )
    return


@app.cell
def _(P_test, P_test_range_labels, X_test, explain, explanation, smiles_df):
    high_mask = P_test_range_labels.reindex(X_test.index).eq("High").to_numpy()
    X_test_high = X_test.loc[P_test_range_labels == "High"]
    P_test_high = P_test.loc[P_test_range_labels == "High"]
    #slice the Explanation

    expl_high = explanation[high_mask]          # works in recent SHAP (row slicing)

    explain.shap_mol_fp(explanation=expl_high,
                X_test=X_test_high, 
                P_test=P_test_high, 
                title="Predicted PCE Range (High)", 
                smiles_df=smiles_df,
                plot_type='dot', 
                max_display=10
    )
    return


if __name__ == "__main__":
    app.run()
