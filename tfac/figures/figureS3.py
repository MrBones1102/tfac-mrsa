"""
Creates Figure S3 -- Alternate Model Performance
"""
import matplotlib
from matplotlib.lines import Line2D
import numpy as np
from os.path import abspath, dirname, join
import pandas as pd
from sklearn.metrics import r2_score, roc_curve

from tfac.figures.common import getSetup
from tfac.cmtf import OPTIMAL_RANK
from tfac.dataImport import import_validation_patient_metadata, get_factors
from tfac.predict import get_accuracy, predict_known, predict_validation, \
    predict_regression

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][2:]
PATH_HERE = dirname(dirname(abspath(__file__)))


def get_predictions(data_types, patient_data):
    """
    Predicts validation samples.

    Parameters:
        data_types (list[tuple]): data sources to predict
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    predictions = pd.DataFrame(
        index=patient_data.index
    )
    probabilities = predictions.copy()

    val_predictions = predictions.copy(deep=True)
    val_predictions = val_predictions.loc[patient_data['status'] == 'Unknown']
    val_probabilities = val_predictions.copy(deep=True)

    validation_meta = import_validation_patient_metadata()

    for data_type in data_types:
        for svc in [True, False]:
            if data_type == 'CMTF' and svc:
                continue

            source = data_type[0]
            data = data_type[1]

            if svc:
                source = source + ': SVM'
            else:
                source = source + ': LR'

            data = data.reindex(index=patient_data.index)
            data = data.dropna(axis=0)
            labels = patient_data.loc[data.index, 'status']

            _predictions = predict_validation(data, labels, svc=svc)
            _probabilities = predict_validation(
                data,
                labels,
                svc=svc,
                predict_proba=True
            )
            val_predictions.loc[_predictions.index, source] = _predictions
            val_probabilities.loc[_probabilities.index, source] = _probabilities

            _predictions, _ = predict_known(data, labels, svc=svc)
            _probabilities, _ = predict_known(
                data,
                labels,
                method='predict_proba',
                svc=svc
            )
            predictions.loc[_predictions.index, source] = _predictions
            probabilities.loc[_probabilities.index, source] = _probabilities

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']
    val_predictions.loc[:, 'Actual'] = validation_meta.loc[:, 'status']
    return predictions, probabilities, val_predictions, val_probabilities


def get_accuracies(samples):
    """
    Calculates prediction accuracy for samples with known outcomes.

    Parameters:
        samples (pandas.DataFrame): predictions for different models

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    actual = samples.loc[:, 'Actual']
    samples = samples.drop(['Actual'], axis=1)

    d_types = samples.columns
    accuracies = pd.Series(
        index=d_types,
        dtype=float
    )

    for d_type in d_types:
        col = samples.loc[:, d_type]
        col = col.dropna()
        accuracies.loc[d_type] = get_accuracy(col, actual)

    return accuracies


def plot_results(predictions, probabilities, val_predictions,
                 val_probabilities):
    """
    Plots prediction model performance.

    Parameters:
        predictions (pandas.DataFrame): predictions for training samples
        probabilities (pandas.DataFrame): predicted probability of persistence
            for training samples
        val_predictions (pandas.DataFrame): predictions for validation cohort
            samples
        val_probabilities (pandas.DataFrame): predicted probability of
            persistence for validation samples

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (5, 5)
    layout = {
        'ncols': 2,
        'nrows': 2
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # Cross-validation Accuracies

    predictions = predictions.loc[predictions['Actual'] != 'Unknown']
    accuracies = get_accuracies(predictions)
    axs[0].bar(
        np.arange(len(accuracies)),
        accuracies,
        width=1,
        color=COLOR_CYCLE[:len(accuracies)],
        label=accuracies.index
    )

    axs[0].set_ylim(0, 1)
    axs[0].legend()
    # ticks = [label.replace(' ', '\n') for label in accuracies.index]
    axs[0].set_xticks(
        np.arange(0, len(accuracies))
    )
    axs[0].set_xticklabels(
        # ticks,
        accuracies.index,
        fontsize=9,
        ma='right',
        rotation=90,
        va='top'
    )
    axs[0].set_ylabel('Prediction Accuracy')

    # AUC-ROC Curves

    legend_lines = []
    legend_names = []
    for d_type, color in zip(probabilities.columns,
                             COLOR_CYCLE[:probabilities.shape[1]]):
        col = probabilities.loc[:, d_type].dropna()
        actual = predictions.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        axs[1].plot(fpr, tpr, color=color, linestyle='--')

        legend_lines.append(Line2D([0], [0], color=color, linewidth=0.5))
        legend_names.append(d_type)

    axs[1].set_xlim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[1].set_xlabel('False Positive Rate')
    axs[1].set_ylabel('True Positive Rate')
    axs[1].legend(legend_lines, legend_names)
    axs[1].plot([0, 1], [0, 1], color='k', linestyle='--')

    # Validation Accuracies

    val_accuracies = get_accuracies(val_predictions)
    axs[2].bar(
        np.arange(len(val_accuracies)),
        val_accuracies,
        width=1,
        color=COLOR_CYCLE[:len(val_accuracies)],
        label=val_accuracies.index
    )

    axs[2].set_ylim(0, 1)
    axs[2].legend()
    axs[2].set_xticks(np.arange(len(val_accuracies)))
    axs[2].set_xticklabels(
        # ticks,
        val_accuracies.index,
        fontsize=9,
        ma='right',
        rotation=90,
        va='top'
    )
    axs[2].set_ylabel('Prediction Accuracy')

    # Validation AUC-ROC Curves

    for d_type, color in zip(val_probabilities.columns,
                             COLOR_CYCLE[:val_probabilities.shape[1]]):
        col = val_probabilities.loc[:, d_type].dropna()
        actual = val_predictions.loc[col.index, 'Actual'].astype(int)
        fpr, tpr, _ = roc_curve(actual, col)
        axs[3].plot(fpr, tpr, color=color, linestyle='--')

    axs[3].set_xlim(0, 1)
    axs[3].set_ylim(0, 1)
    axs[3].legend(legend_lines, legend_names)
    axs[3].set_xlabel('False Positive Rate')
    axs[3].set_ylabel('True Positive Rate')
    axs[3].plot([0, 1], [0, 1], color='k', linestyle='--')

    return fig


def makeFigure():
    tfac, pca, patient_data = get_factors()
    components = tfac.factors[0]

    data_types = [
        (
            'CMTF',
            pd.DataFrame(
                components,
                index=patient_data.index,
                columns=list(range(1, components.shape[1] + 1))
            )
        ),
        (
            'PCA',
            pd.DataFrame(
                pca.scores,
                index=patient_data.index,
                columns=list(range(1, OPTIMAL_RANK + 1))
            )
        ),
        # (
        #     'PCA-imputed',
        #     pd.DataFrame(
        #         pca.projection,
        #         index=patient_data.index
        #     )
        # )
    ]

    predictions, probabilities, val_predictions, val_probabilities = \
        get_predictions(data_types, patient_data)

    fig = plot_results(
        predictions,
        probabilities,
        val_predictions,
        val_probabilities
    )

    return fig
