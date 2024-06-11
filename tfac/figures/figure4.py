"""
Creates Figure 4 -- Model Interpretation
"""
from os.path import abspath, dirname
from string import ascii_uppercase

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.utils import resample

from tfac.figures.common import getSetup
from tfac.dataImport import form_tensor, import_cytokines, get_factors, \
    reorder_table
from tfac.predict import run_model, predict_regression

N_BOOTSTRAP = 30
PATH_HERE = dirname(dirname(abspath(__file__)))
TARGETS = ['status', 'gender']


def bootstrap_weights(components):
    """
    Predicts samples with unknown outcomes.

    Parameters:
        None

    Returns:
        weights (pandas.DataFrame): mean and StD of component weights w/r to
            prediction targets
    """
    _, _, patient_data = form_tensor()
    patient_data = patient_data.reset_index(drop=True)
    patient_data = patient_data.loc[patient_data['status'] != 'Unknown']

    components = components[1][0]
    components = components[patient_data.index, :]

    stats = ['Mean', 'StD']
    index = pd.MultiIndex.from_product([TARGETS, stats])
    weights = pd.DataFrame(
        index=index,
        columns=list(range(1, components.shape[1] + 1))
    )

    for target in TARGETS:
        coef = []
        for sample in range(N_BOOTSTRAP):
            data, labels = resample(components, patient_data.loc[:, target])
            if target == 'age':
                _, _coef = predict_regression(data, labels)
            else:
                _, _, _coef = run_model(data, labels, return_coef=True)

            coef.append(_coef)

        coef = scale(coef, axis=1)
        weights.loc[(target, 'Mean'), :] = np.mean(coef, axis=0)
        weights.loc[(target, 'StD'), :] = np.std(coef, axis=0, ddof=1)

    return weights


def tfac_setup():
    """
    Import cytokine data and correlate tfac components to cytokines and
    data sources.

    Parameters:
        None

    Returns:
        subjects (pandas.DataFrame): patient correlations to tfac components
        cytos (pandas.DataFrame): cytokine correlations to tfac components
        source (pandas.DataFrame): cytokine source correlations to tfac
            components
        pat_info (pandas.DataFrame): patient meta-data
    """
    factors, _, pat_info = get_factors()
    plasma, _ = import_cytokines()
    cytokines = plasma.index

    pat_info.loc[:, 'sorted'] = range(pat_info.shape[0])
    pat_info = pat_info.sort_values(['cohort', 'type', 'status'])
    sort_idx = pat_info.loc[:, 'sorted']
    pat_info = pat_info.drop('sorted', axis=1)
    pat_info = pat_info.T

    col_names = [f"Cmp. {i}" for i in np.arange(1, factors.rank + 1)]
    subjects = pd.DataFrame(
        factors.factors[0][sort_idx, :],
        columns=col_names,
        index=[str(x) for x in pat_info.columns]
    )
    cytos = pd.DataFrame(
        factors.factors[1],
        columns=col_names,
        index=cytokines
    )
    source = pd.DataFrame(
        factors.factors[2],
        columns=col_names,
        index=["Serum", "Plasma"]
    )
    cytos = reorder_table(cytos)

    return subjects, cytos, source, pat_info, factors


def plot_results(weights, subjects, cytos, source, pat_info):
    """
    Plots component weights and interpretation.

    Parameters:
        weights (pandas.DataFrame): mean and StD of component weights w/r to
            prediction targets
        subjects (pandas.DataFrame): patient correlations to tfac components
        cytos (pandas.DataFrame): cytokine correlations to tfac components
        source (pandas.DataFrame): cytokine source correlations to tfac
            components
        pat_info (pandas.DataFrame): patient meta-data
    """
    fig_size = (5, 5)
    layout = {
        # 'height_ratios': [1, 0.5],
        'hspace': 0.3,
        'ncols': 1,
        'nrows': 2,
        'wspace': 0
    }
    _, fig, gs = getSetup(
        fig_size,
        layout,
        style=None
    )
    top_gs = gs[0].subgridspec(
        ncols=3,
        nrows=1,
        width_ratios=[35, 4, 25]
    )
    bottom_gs = gs[1].subgridspec(
        ncols=14,
        nrows=1,
        width_ratios=[25, 12, 1, 5, 1, 5, 1, 1, 1, 1, 1, 25, 1, 1],
        wspace=0
    )

    for gs in [top_gs, bottom_gs]:
        for col in range(gs.ncols):
            fig.add_subplot(gs[col])

    fig.delaxes(fig.axes[0])
    fig.delaxes(fig.axes[0])
    axs = fig.axes
    for ax in axs:
        ax.set_frame_on(False)
    axs[0].set_frame_on(True)

    spacers = [1, 4, 5, 6, 8, 10, 16]
    for spacer in spacers:
        axs[spacer].set_xticks([])
        axs[spacer].set_yticks([])

    # Determine scale
    vmin = min(subjects.values.min(), cytos.values.min(), source.values.min())
    vmax = max(subjects.values.max(), cytos.values.max(), source.values.max())

    # Plot main graphs
    sns.heatmap(
        subjects,
        cmap="PRGn",
        center=0,
        xticklabels=True,
        yticklabels=False,
        cbar_ax=axs[15],
        vmin=vmin,
        vmax=vmax,
        ax=axs[14]
    )
    sns.heatmap(
        cytos,
        cmap="PRGn",
        center=0,
        yticklabels=True,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        ax=axs[2]
    )
    sns.heatmap(
        source,
        cmap="PRGn",
        center=0,
        yticklabels=True,
        cbar=False,
        vmin=vmin,
        vmax=vmax,
        ax=axs[3]
    )
    axs[2].set_yticklabels(cytos.index, fontsize=7)
    axs[3].set_yticklabels(["Serum", "Plasma"], rotation=0)
    axs[3].set_xticks(np.arange(0.5, source.shape[1]))
    axs[3].set_xticklabels([f'Cmp. {i}' for i in range(1, source.shape[1] + 1)])

    # Set up subject colorbars
    outcome_colors = ["gray", "lightgreen", "brown"]
    cohort_colors = ["deeppink", "orchid", "pink"]
    type_colors = \
        ["black", "orange", "purple", "green", "red", "yellow", "blue"]
    outcome_cmap = sns.color_palette(outcome_colors)
    cohort_cmap = sns.color_palette(cohort_colors)
    type_cmap = sns.color_palette(type_colors)

    # Data types bar
    types = pd.DataFrame(pat_info.loc["type"]).set_index("type")
    types["Type"] = 0
    types[types.index == "0Serum"] = 6
    types[types.index == "1Plasma"] = 5
    types[types.index == "2RNAseq"] = 4
    types[types.index == "0Serum1Plasma"] = 3
    types[types.index == "0Serum2RNAseq"] = 2
    types[types.index == "1Plasma2RNAseq"] = 1
    types[types.index == "0Serum1Plasma2RNAseq"] = 0

    sns.heatmap(
        types,
        ax=axs[11],
        cbar_ax=axs[5],
        yticklabels=False,
        xticklabels=False,
        cmap=type_cmap
    )
    colorbar = axs[11].collections[0].colorbar
    colorbar.set_ticks(np.linspace(.4, 5.6, 7))
    axs[5].set_yticklabels(
        [
            "All types", "Plasma\nRNAseq", "Serum\nRNAseq", "Serum\nPlasma",
            "RNAseq", "Plasma", "Serum"],
        va="center",
        rotation=0
    )
    axs[5].yaxis.set_ticks_position('left')
    axs[11].set_ylabel("")

    # Cohort bar
    cohort = pd.DataFrame(pat_info.loc["cohort"]).set_index("cohort")
    cohort["Cohort"] = 0
    cohort[cohort.index == 1] = 2
    cohort[cohort.index == 2] = 1
    cohort[cohort.index == 3] = 0

    sns.heatmap(
        cohort,
        ax=axs[12],
        cbar_ax=axs[7],
        yticklabels=False,
        xticklabels=False,
        cmap=cohort_cmap
    )
    colorbar = axs[12].collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    axs[7].yaxis.set_ticklabels(
        ["Cohort 3", "Cohort 2", "Cohort 1"],
        va="center")
    axs[7].yaxis.set_ticks_position('left')
    axs[7].yaxis.set_tick_params(rotation=90)
    axs[12].set_ylabel("")

    # Outcome bar
    outs = pd.DataFrame(pat_info.loc["status"]).set_index("status")
    outs["Outcome"] = 0
    outs[outs.index == "0"] = 1
    outs[outs.index == "1"] = 2
    outs[outs.index == "Unknown"] = 0

    sns.heatmap(
        outs,
        ax=axs[13],
        cbar_ax=axs[9],
        yticklabels=False,
        xticklabels=False,
        cmap=outcome_cmap
    )
    colorbar = axs[13].collections[0].colorbar
    colorbar.set_ticks([.33, 1, 1.66])
    axs[9].yaxis.set_ticklabels(
        ["Unknown", "Resolver", "Persister"],
        va='center'
    )
    axs[9].yaxis.set_ticks_position('left')
    axs[9].yaxis.set_tick_params(rotation=90)
    axs[13].set_ylabel("")

    for ii, ax in enumerate([axs[0], axs[2], axs[3], axs[14]]):
        ax.text(
            -0.2,
            1.1,
            ascii_uppercase[ii],
            transform=ax.transAxes,
            fontsize=14,
            fontweight="bold",
            va="top"
        )

    for offset, target in enumerate(TARGETS):
        axs[0].errorbar(
            weights.loc[(target, 'Mean')],
            range(
                offset,
                (len(TARGETS) + 5) * weights.shape[1],
                len(TARGETS) + 5
            ),
            marker='.',
            xerr=weights.loc[(target, 'StD')],
            linestyle='',
            capsize=2
        )

    axs[0].legend(
        ['Persistence', 'Sex']
    )

    axs[0].plot(
        [0, 0],
        [-100, 100],
        color='k',
        linestyle='--'
    )

    axs[0].set_xlabel('Model Coefficient')
    axs[0].set_xlim(-3, 3)
    axs[0].set_ylim(-3, 53)
    axs[0].set_yticks(np.arange(0.5, 53, 7))
    axs[0].set_yticklabels(
        [f'Cmp. {i}' for i in range(1, 9)]
    )

    plt.subplots_adjust(left=0.1, right=0.925, top=0.95)

    return fig


def makeFigure():
    subjects, cytos, source, pat_info, factors = tfac_setup()
    weights = bootstrap_weights(factors)
    return plot_results(weights, subjects, cytos, source, pat_info)
