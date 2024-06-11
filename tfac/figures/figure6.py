import re

import numpy as np
import pandas as pd

from tfac.figures.common import getSetup
from tfac.dataImport import get_factors
from tfac.predict import predict_known


def makeFigure():
    predicted = get_predictions()
    accuracies = get_accuracy_by_dtype(predicted)
    fig = plot_results(accuracies)

    return fig


def get_accuracy_by_dtype(predictions):
    """
    Calculates model accuracy w/r to available data types.

    Parameters:
        predictions (pandas.Series): model predictions for each sample

    Returns:
        accuracies (pandas.Series): model accuracy w/r to data types
    """
    d_types = list(set(predictions['Type']))
    accuracies = pd.Series(
        index=d_types,
        dtype=float
    )

    for d_type in d_types:
        predicted = predictions.loc[predictions['Type'] == d_type]
        accuracies.loc[d_type] = np.mean(predicted['Correct'])

    accuracies = accuracies.dropna()
    accuracies = accuracies.sort_values(ascending=False)
    return accuracies


def plot_results(accuracies):
    """
    Plots model accuracy relative to different data sources.

    Parameters:
        accuracies (pandas.Series): model accuracy w/r to data types

    Returns:
        fig (matplotlib.Figure): bar plot depicting model accuracy w/r to data
            types
    """
    fig_size = (4, 4)
    layout = {
        'ncols': 1,
        'nrows': 1
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    labels = [re.sub(r'\d', '/', d_type[1:]) for d_type in accuracies.index]
    axs[0].bar(range(len(accuracies)), accuracies)
    axs[0].set_xticks(range(len(accuracies)))
    axs[0].set_xticklabels(labels, rotation=45, ha='right', fontsize=10)
    axs[0].set_ylabel('Mean Accuracy', fontsize=12)
    axs[0].set_xlabel('Datatypes Available', fontsize=12)

    return fig


def get_predictions():
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        None

    Returns:
        predictions (pandas.Series): model predictions for each sample
    """
    components, _, patient_data = get_factors()
    patient_data = patient_data.loc[:, ['status', 'type']]
    components = components[1][0]

    data = pd.DataFrame(
        components,
        index=patient_data.index,
        columns=list(range(1, components.shape[1] + 1))
    )
    predicted = pd.DataFrame(
        index=patient_data.index
    )
    labels = patient_data.loc[:, 'status']

    predictions, _ = predict_known(data, labels)
    for i in predictions.index:
        if predictions.loc[i] == labels.loc[i]:
            predicted.loc[i, 'Correct'] = 1
        else:
            predicted.loc[i, 'Correct'] = 0

    predicted.loc[:, 'Type'] = patient_data.loc[:, 'type']
    return predicted
