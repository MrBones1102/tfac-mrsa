"""
Creates Figure 5 -- Reduced Model
"""
import matplotlib
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from os.path import abspath, dirname
import pandas as pd
from sklearn.metrics import roc_curve

from tfac.figures.common import getSetup
from tfac.dataImport import import_validation_patient_metadata, get_factors, \
    import_cytokines, import_rna
from tfac.predict import get_accuracy, predict_known

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']
PATH_HERE = dirname(dirname(abspath(__file__)))
PERSISTENCE_COMPONENTS = [1, 2, 4, 6]


def run_cv(components, patient_data, svc=True):
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        components (numpy.array): CMTF components
        patient_data (pandas.DataFrame): patient metadata
        svc (bool, default: True): use svc for classification

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    labels = patient_data.loc[components.index, 'status'].astype(int)

    predictions = pd.DataFrame(
        index=patient_data.index
    )
    probabilities = predictions.copy()

    predictions.loc[:, 'Full'], _ = predict_known(components, labels, svc=svc)
    probabilities.loc[:, 'Full'], _ = predict_known(
        components,
        labels,
        method='predict_proba',
        svc=svc
    )

    predictions.loc[:, '1, 2, 4 & 6'], _ = predict_known(
        components.loc[:, PERSISTENCE_COMPONENTS],
        labels,
        svc=svc
    )
    probabilities.loc[:, '1, 2, 4 & 6'], _ = predict_known(
        components.loc[:, PERSISTENCE_COMPONENTS],
        labels,
        svc=svc,
        method='predict_proba'
    )

    summed = pd.concat(
        [
            components.loc[:, 2],
            components.loc[:, [4, 6]].sum(axis=1)
        ],
        axis=1
    )

    predictions.loc[:, '2, 4 + 6'], model = predict_known(
        summed,
        labels,
        svc=svc
    )
    probabilities.loc[:, '2, 4 + 6'], _ = predict_known(
        summed,
        labels,
        svc=svc,
        method='predict_proba'
    )

    for i in np.arange(len(PERSISTENCE_COMPONENTS)):
        reduced = PERSISTENCE_COMPONENTS[:i] + PERSISTENCE_COMPONENTS[i+1:]
        name = ', '.join([str(i) for i in reduced[:-1]]) + f' & {reduced[-1]}'
        predictions[name], _ = predict_known(
            components.loc[:,  reduced],
            labels,
            svc=svc
        )
        probabilities[name], _ = \
            predict_known(
                components.loc[:, reduced],
                labels,
                method='predict_proba',
                svc=svc
            )

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions.astype(int), probabilities, model


def get_accuracies(samples):
    """
    Calculates prediction accuracy for samples with known outcomes.

    Parameters:
        samples (pandas.DataFrame): predictions for different models

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    actual = samples.loc[:, 'Actual']
    samples = samples.drop('Actual', axis=1)

    d_types = samples.columns
    accuracies = pd.Series(
        index=d_types,
        dtype=float
    )

    for d_type in d_types:
        col = samples[d_type]
        accuracies[d_type] = get_accuracy(col, actual)

    return accuracies


def plot_results(predictions, probabilities, lr_predictions, model, components,
                 t_fac, patient_data):
    """
    Plots prediction model performance.

    Parameters:
        predictions (pandas.DataFrame): predictions for training samples
        probabilities (pandas.DataFrame): predicted probability of
            persistence for training samples
        lr_predictions (pandas.DataFrame): LR predictions for training samples
        model (sklearn.SVM): Best reduced SVM model
        components (pandas.DataFrame): CMTF components
        t_fac (CPTensor): CMTF factorization result
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (6, 2)
    layout = {
        'ncols': 3,
        'nrows': 1,
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # Cross-validation Accuracies

    accuracies = get_accuracies(predictions)
    lr_accuracies = get_accuracies(lr_predictions)
    axs[0].bar(
        np.arange(0, 5 * len(accuracies), 5),
        accuracies,
        color=COLOR_CYCLE[:len(accuracies)],
        width=2,
    )
    axs[0].bar(
        np.arange(2, 5 * len(lr_accuracies), 5),
        lr_accuracies,
        color=COLOR_CYCLE[:len(accuracies)],
        width=2,
        hatch='//',
    )
    axs[0].legend(
        [
            Patch(facecolor='black', edgecolor='white'),
            Patch(facecolor='black', edgecolor='white', hatch='//')
        ],
        [
            'SVM',
            'LR'
        ],
        handleheight=1,
        handlelength=3
    )

    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(
        np.arange(1, 5 * len(accuracies), 5)
    )

    labels = accuracies.index
    axs[0].set_xticklabels(
        labels,
        rotation=45,
        ha='right',
        va='top'
    )

    axs[0].set_xlabel('Components')
    axs[0].set_ylabel('Balanced Accuracy')

    # AUC-ROC Curves

    for i, reduced in enumerate(probabilities.columns):
        fpr, tpr, _ = roc_curve(
            predictions.loc[:, 'Actual'],
            probabilities[reduced]
        )

        axs[1].plot(fpr, tpr, color=COLOR_CYCLE[i])

    axs[1].set_xticks(np.linspace(0, 1, 6))
    axs[1].set_yticks(np.linspace(0, 1, 6))
    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(labels)
    axs[1].plot([0, 1], [0, 1], color='k', linestyle='--')

    # Best Model Scatter

    color = patient_data.loc[:, 'status'].astype(int)
    color = color.replace(0, 'blue')
    color = color.replace(1, 'red')

    style = patient_data.loc[:, 'gender'].astype(int)
    style = style.replace(0, 's')
    style = style.replace(1, 'o')

    xx, yy = np.meshgrid(
        np.linspace(-1.1, 1.1, 23),
        np.linspace(-1.1, 1.1, 23)
    )
    grid = np.c_[xx.ravel(), yy.ravel()]
    prob_map = model.predict_proba(grid)[:, 1].reshape(xx.shape)

    axs[2].contourf(
        xx,
        yy,
        prob_map,
        cmap='coolwarm',
        linestyles='--'
    )

    for marker in ['s', 'o']:
        index = style.loc[style == marker].index
        axs[2].scatter(
            components.loc[index, 2],
            components.loc[index, [4, 6]].sum(axis=1),
            c=color.loc[index],
            s=10,
            edgecolors='k',
            marker=marker,
            zorder=2
        )

    axs[2].set_xticks(np.arange(-1, 1.1, 0.5))
    axs[2].set_xlim([-1.1, 1.1])
    axs[2].set_yticks(np.arange(-1, 1.1, 0.5))
    axs[2].set_ylim([-1.1, 1.1])

    axs[2].set_xlabel(f'Component 2')
    axs[2].set_ylabel(f'Components 4 + 6')

    legend_markers = [
        Line2D([0], [0], lw=4, color='blue', label='Resolver'),
        Line2D([0], [0], lw=4, color='red', label='Persister'),
        Line2D([0], [0], marker='s', color='k', label='Male'),
        Line2D([0], [0], marker='o', color='k', label='Female')
    ]
    axs[2].legend(handles=legend_markers)

    return fig


def makeFigure():
    t_fac, _, patient_data = get_factors()
    val_data = import_validation_patient_metadata()
    patient_data.loc[val_data.index, 'status'] = val_data.loc[:, 'status']

    components = t_fac[1][0]
    components = pd.DataFrame(
        components,
        index=patient_data.index,
        columns=list(np.arange(1, components.shape[1] + 1))
    )

    predictions, probabilities, model = run_cv(components, patient_data)
    lr_predictions, lr_probabilities, _ = run_cv(
        components,
        patient_data,
        svc=False
    )

    fig = plot_results(
        predictions,
        probabilities,
        lr_predictions,
        model,
        components,
        t_fac,
        patient_data
    )

    return fig
