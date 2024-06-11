"""
Creates Figure S5 -- PCA Model Performance Comparison
"""
import matplotlib
import numpy as np
from os.path import abspath, dirname
import pandas as pd

from tfac.figures.common import getSetup
from tfac.dataImport import import_validation_patient_metadata, get_factors, import_cytokines, import_rna
from tfac.predict import get_accuracy, predict_known, predict_validation, predict_regression

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color'][2:]
PATH_HERE = dirname(dirname(abspath(__file__)))


def run_validation(data_types, patient_data):
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
    predictions = predictions.loc[patient_data['status'] == 'Unknown']
    validation_meta = import_validation_patient_metadata()

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]

        data = data.reindex(index=patient_data.index)
        data = data.dropna(axis=0)
        labels = patient_data.loc[data.index, 'status']

        _predictions = predict_validation(data, labels)
        _probabilities = predict_validation(data, labels, predict_proba=True)
        predictions.loc[_predictions.index, source] = _predictions

    predictions.loc[:, 'Actual'] = validation_meta.loc[:, 'status']
    return predictions


def run_cv(data_types, patient_data):
    """
    Predicts samples with known outcomes via cross-validation.

    Parameters:
        data_types (list[tuple]): data sources to predict
        patient_data (pandas.DataFrame): patient metadata

    Returns:
        predictions (pandas.Series): predictions for each data source
    """
    predictions = pd.DataFrame(
        index=patient_data.index
    )

    for data_type in data_types:
        source = data_type[0]
        data = data_type[1]

        data = data.reindex(index=patient_data.index)
        data = data.dropna(axis=0)
        labels = patient_data.loc[data.index, 'status']

        _predictions, _ = predict_known(data, labels)
        predictions.loc[_predictions.index, source] = _predictions

    predictions.loc[:, 'Actual'] = patient_data.loc[:, 'status']

    return predictions


def run_age_regression(data, patient_data):
    """
    Predicts patient age from provided data.

    Parameters:
        data (pandas.DataFrame): data to predict age
        patient_data (pandas.DataFrame): patient metadata, including age
    """
    labels = patient_data.loc[:, 'age']
    age_predictions, _ = predict_regression(data, labels)
    age_predictions.name = 'CMTF'

    age_predictions = pd.DataFrame(age_predictions)
    age_predictions.loc[:, 'Actual'] = labels.loc[age_predictions.index]

    return age_predictions


def get_accuracies(samples):
    """
    Calculates prediction accuracy for samples with known outcomes.

    Parameters:
        samples (pandas.DataFrame): predictions for different models

    Returns:
        accuracies (pandas.Series): model accuracy w/r to each model type
    """
    cmtf = samples.loc[:, 'CMTF']
    pca = samples.loc[:, 'PCA']
    actual = samples.loc[:, 'Actual']
    samples = samples.drop(['CMTF', 'PCA', 'Actual'], axis=1)

    d_types = samples.columns
    accuracies = pd.Series(
        index=d_types,
        dtype=float
    )
    cmtf_accuracies = pd.Series(
        index=d_types,
        dtype=float
    )
    pca_accuracies = cmtf_accuracies.copy()

    for d_type in d_types:
        col = samples.loc[:, d_type]
        col = col.dropna()
        labels = actual.loc[col.index]
        cmtf_col = cmtf.loc[col.index]
        pca_col = pca.loc[col.index]

        accuracies.loc[d_type] = get_accuracy(col, actual)
        cmtf_accuracies.loc[d_type] = get_accuracy(cmtf_col, labels)
        pca_accuracies.loc[d_type] = get_accuracy(pca_col, labels)

    return accuracies, cmtf_accuracies, pca_accuracies


def plot_results(train_samples, validation_samples):
    """
    Plots prediction model performance.

    Parameters:
        train_samples (pandas.DataFrame): predictions for training samples
        validation_samples (pandas.DataFrame): predictions for validation
            cohort samples

    Returns:
        fig (matplotlib.Figure): figure depicting predictions for all samples
    """
    fig_size = (5, 2.5)
    layout = {
        'ncols': 2,
        'nrows': 1,
    }
    axs, fig, _ = getSetup(
        fig_size,
        layout
    )

    # Cross-validation Accuracies

    train_samples = train_samples.loc[train_samples['Actual'] != 'Unknown']
    accuracies, train_cmtf, train_pca = get_accuracies(train_samples)
    axs[0].bar(
        np.arange(0, 4 * len(accuracies), 4),
        accuracies,
        width=1
    )
    axs[0].bar(
        np.arange(1, 4 * len(accuracies), 4),
        train_pca,
        width=1
    )
    axs[0].bar(
        np.arange(2, 4 * len(train_cmtf), 4),
        train_cmtf,
        width=1
    )

    axs[0].set_xlim(-1, 4 * len(accuracies) - 1)
    axs[0].set_ylim(0, 1)
    axs[0].legend(['Raw Data', 'PCA', 'CMTF'])
    ticks = [label.replace(' ', '\n') for label in accuracies.index]
    axs[0].set_xticks(
        np.arange(0.5, 4 * len(accuracies), 4)
    )
    axs[0].set_xticklabels(
        ticks,
        fontsize=9,
        ma='right',
        rotation=90,
        va='top'
    )
    axs[0].set_ylabel('Prediction Accuracy')
    axs[0].text(
        -0.35,
        0.9,
        'A',
        fontsize=14,
        fontweight='bold',
        transform=axs[0].transAxes
    )

    # Validation Accuracies

    val_accuracies, val_cmtf, val_pca = get_accuracies(validation_samples)
    axs[1].bar(
        np.arange(0, 4 * len(val_accuracies), 4),
        val_accuracies,
        width=1
    )
    axs[1].bar(
        np.arange(1, 4 * len(val_accuracies), 4),
        val_pca,
        width=1
    )
    axs[1].bar(
        np.arange(2, 4 * len(val_cmtf), 4),
        val_cmtf,
        width=1
    )

    axs[1].set_xlim(-1, 4 * len(accuracies) - 1)
    axs[1].set_ylim(0, 1)
    axs[1].legend(['Raw Data', 'PCA', 'CMTF'])
    axs[1].set_xticks(
        np.arange(0.5, 4 * len(val_accuracies), 4)
    )
    axs[1].set_xticklabels(
        ticks,
        fontsize=9,
        ma='right',
        rotation=90,
        va='top'
    )
    axs[1].set_ylabel('Prediction Accuracy')
    axs[1].text(
        -0.35,
        0.9,
        'C',
        fontsize=14,
        fontweight='bold',
        transform=axs[1].transAxes
    )

    return fig


def makeFigure():
    plasma_cyto, serum_cyto = import_cytokines()
    rna = import_rna()
    components, pcaFac, patient_data = get_factors()
    components = components[1][0]
    pca_components = pcaFac.scores

    data_types = [
        ('Plasma Cytokines', plasma_cyto.T),
        ('Plasma IL-10', plasma_cyto.loc['IL-10', :]),
        ('Serum Cytokines', serum_cyto.T),
        ('Serum IL-10', serum_cyto.loc['IL-10', :]),
        ('RNA Modules', rna),
        ('PCA', pd.DataFrame(
            pca_components,
            index=patient_data.index,
            columns=list(range(1, components.shape[1] + 1))
        )),
        ('CMTF', pd.DataFrame(
            components,
            index=patient_data.index,
            columns=list(range(1, components.shape[1] + 1))
        ))
    ]

    validation_samples = run_validation(data_types, patient_data)
    train_samples = run_cv(data_types, patient_data)

    fig = plot_results(train_samples, validation_samples)

    return fig
