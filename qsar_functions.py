import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import plotly.express as px

import molplotly

from sklearn.metrics import r2_score, mean_absolute_error, root_mean_squared_error

def reg_results_df(trained_model, X_train, X_test, Y_train, Y_test):

    # gather predictions
    P_train = trained_model.predict(X_train)
    P_test = trained_model.predict(X_test)

    # add train-test labels for combined plot
    train_labels = ['Train'] * len(P_train)
    test_labels = ['Test'] * len(P_test)

    # store in dict for transform to df with appropriate indexes
    train_dict = {
        'Measured': Y_train.values,
        'Predicted': P_train,
        'Dataset': train_labels
    }

    test_dict = {
        'Measured': Y_test.values,
        'Predicted': P_test,
        'Dataset': test_labels
    }

    # dataframed results
    train_df = pd.DataFrame.from_dict(train_dict)
    train_df = train_df.set_index(X_train.index)

    test_df = pd.DataFrame.from_dict(test_dict)
    test_df = test_df.set_index(X_test.index)

    comb_df = pd.concat([train_df, test_df])

    return comb_df, train_dict, test_dict


# published 
def reg_results_viz_sns(comb_df, train_dict, test_dict, box_adjustment=0.2, title=''):
     # ---- train results ----
    Y_train = train_dict['Measured']
    P_train = train_dict['Predicted']
    plot_range = [Y_train.min()-1, Y_train.max()+1]
    r2 = r2_score(Y_train, P_train) # precision, recall, 
    train_mae = mean_absolute_error(Y_train, P_train)
    train_rmse = root_mean_squared_error(Y_train, P_train)

    # ---- test results ----
    Y_test = test_dict['Measured']
    P_test = test_dict['Predicted'] 
    q2 = r2_score(Y_test, P_test)
    test_mae = mean_absolute_error(Y_test, P_test)
    test_rmse = root_mean_squared_error(Y_test, P_test)

    fig, ax = plt.subplots()
    ax.set_title(f"{title}")
    # set this outside instead
    # sns.set_style("whitegrid")
    sns.scatterplot(ax=ax, data=comb_df, x='Predicted', y='Measured', hue='Dataset', legend=False, palette='colorblind')
    ax.set_title(f"{title}")
    ax.set_xlim(plot_range)
    ax.set_ylim(plot_range)
    
    # plot diagonal
    diag_arr = np.arange(plot_range[0], plot_range[1], 0.5)

    r2_str = ('%.2f' %r2)
    q2_str = ('%.2f' %q2)

    train_mae_str = ('%.2f' % train_mae)
    test_mae_str = ('%.2f' % test_mae)

    train_rmse_str = ('%.2f' % train_rmse)
    test_rmse_str = ('%.2f' % test_rmse)

    train_box_str = f"$R^2$ = {r2_str}\n" + \
                    "$MAE_{train}$" + \
                    f" = {train_mae_str}\n" + \
                    "$RMSE_{train}$" + \
                    f" = {train_rmse_str}"
    
    test_box_str =  f"$Q^2$ = {q2_str}\n" + \
                    "$MAE_{test}$" + \
                    f" = {test_mae_str}\n" + \
                    "$RMSE_{test}$" + \
                    f" = {test_rmse_str}"
    
    ax.plot(diag_arr, diag_arr, 'k--')

    pal = sns.color_palette('colorblind')
    
    placement_down = plot_range[0] + (plot_range[1] - plot_range[1]*(box_adjustment))
    placement_right = plot_range[1]*(box_adjustment)
    
    ax.text(
        placement_right,
        placement_down,
        train_box_str,
        bbox={
            'facecolor': pal[0],
            'alpha': 0.5
            # ,'pad': padding
        },
        horizontalalignment = 'center'
    )

    placement_up = plot_range[1]*(0.5*box_adjustment)
    placement_left = plot_range[0] + (plot_range[1] - plot_range[1]*(1*box_adjustment))
    
    ax.text(
        placement_left,
        placement_up,
        test_box_str,
        bbox={
            'facecolor': pal[1],
            'alpha': 0.5
            # ,'pad': padding
        },
        horizontalalignment = 'center'
    )
    ax.set_aspect('equal', adjustable='box')
    plt.tight_layout()
    return fig, ax
    

def reg_resid_results_viz_sns(comb_df, title=""):
    comb_df['Residuals'] = comb_df['Predicted'] - comb_df['Measured']

    fig, ax = plt.subplots(1, 2, gridspec_kw={'width_ratios':[3,1]}, sharey=True)
    #set outside instead
    # sns.set_style("whitegrid")
    sns.scatterplot(ax=ax[0], data=comb_df, x='Predicted', y='Residuals', hue="Dataset", palette="colorblind")
    ax[0].set(
        xlabel='Predicted Value',
        title=f"{title}"
    )
    ax[0].axhline(
        y=0,
        xmin=0,
        xmax=1,
        color="black",
        linestyle="dashed"
    )

    sns.histplot(ax=ax[1], data=comb_df, y='Residuals', hue='Dataset', palette='colorblind', element='step', legend=False)
    # ax[1].yaxis.set_label_position("right")
    ax[1].set(
        xlabel='Distribution',
        ylabel=None
    )

    ax[1].axhline(
        y=0,
        xmin=0,
        xmax=1,
        color="black",
        linestyle="dashed"
    )

    plt.tight_layout()

def reg_results_smiles(comb_df, smiles_df, smiles_col="SMILES"):
    # get the smiles in the right order
    smiles_split_df = smiles_df[smiles_col].loc[comb_df.index]

    # add them to the measured vs predicted df
    results_smiles_df = pd.concat([comb_df, smiles_split_df], axis=1)

    return results_smiles_df



def reg_results_viz_molplot(comb_df, smiles_df, smiles_col="SMILES", title=''):

    results_smiles = reg_results_smiles(comb_df, smiles_df, smiles_col=smiles_col)

    fig_scatter = px.scatter(
        results_smiles,
        x='Measured',
        y='Predicted',
        color='Dataset',
        title=title,
        labels={
            'Predicted': 'Predicted PCE',
            'Measured': 'Measured PCE',
            'Dataset': 'Dataset'
        },
        width=1200,
        height=800
    )

    Y_true = comb_df["Measured"].values

    fig_scatter.add_shape(
        type="line",
        line=dict(dash='dash'),
        x0 = Y_true.min(),
        y0 = Y_true.min(),
        x1 = Y_true.max(),
        y1 = Y_true.max()
    )

    
    app_scatter = molplotly.add_molecules(
        fig=fig_scatter,
        df=results_smiles,
        smiles_col=smiles_col,
        color_col='Dataset'
    )

    app_scatter.run_server(mode='inline', port=8700, hieght=1000)



