"""
Creates Figure 2 -- CMTF Plotting
"""
import numpy as np
import pandas as pd

from .common import getSetup
from ..dataImport import form_tensor
from ..predict import run_model
from tensorpack import perform_CMTF, calcR2X


def get_r2x_results():
    """
    Calculates CMTF R2X with regards to the number of CMTF components and
    RNA/cytokine scaling.

    Parameters:
        None

    Returns:
        r2x_v_components (pandas.Series): R2X vs. number of CMTF components
        r2x_v_scaling (pandas.Series): R2X vs. RNA/cytokine scaling
    """
    # R2X v. Components
    tensor, matrix, patient_data = form_tensor()
    labels = patient_data.loc[:, 'status']
    components = 12

    r2x_v_components = pd.Series(
        index=np.arange(1, components + 1)
    )
    acc_v_components = pd.Series(
        index=np.arange(1, components + 1).tolist(),
        dtype=float
    )
    for n_components in r2x_v_components.index:
        print(f"Starting decomposition with {n_components} components.")
        t_fac = perform_CMTF(tensor, matrix, r=n_components, tol=1e-9, maxiter=100)
        r2x_v_components.loc[n_components] = t_fac.R2X
        acc_v_components[n_components] = run_model(t_fac.factors[0], labels)[0]

    # R2X v. Scaling
    scalingV = np.logspace(-16, 4, base=2, num=15)
    r2x_v_scaling = pd.DataFrame(
        index=scalingV,
        columns=["Total", "Tensor", "Matrix"]
    )
    acc_v_scaling = pd.Series(
        index=scalingV.tolist(),
        dtype=float
    )
    for scaling in r2x_v_scaling.index:
        tensor, matrix, _ = form_tensor(scaling)
        t_fac = perform_CMTF(tOrig=tensor, mOrig=matrix, tol=1e-9, maxiter=100)
        r2x_v_scaling.loc[scaling, "Total"] = calcR2X(t_fac, tensor, matrix)
        r2x_v_scaling.loc[scaling, "Tensor"] = calcR2X(t_fac, tIn=tensor)
        r2x_v_scaling.loc[scaling, "Matrix"] = calcR2X(t_fac, mIn=matrix)
        acc_v_scaling.loc[scaling] = run_model(t_fac.factors[0], labels)[0]

    return r2x_v_components, acc_v_components, r2x_v_scaling, acc_v_scaling


def plot_results(r2x_v_components, r2x_v_scaling, acc_v_components,
                 acc_v_scaling):
    """
    Plots prediction model performance.

    Parameters:
        r2x_v_components (pandas.Series): R2X vs. number of CMTF components
        r2x_v_scaling (pandas.Series): R2X vs. RNA/cytokine scaling
        acc_v_components (pandas.Series): accuracy vs. number of CMTF components
        acc_v_scaling (pondas.Series): accuracy vs. RNA/cytokine scaling

    Returns:
        fig (matplotlib.Figure): figure depicting CMTF parameterization plots
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

    # R2X v. Components

    axs[0].plot(r2x_v_components.index, r2x_v_components)
    axs[0].set_ylabel('R2X')
    axs[0].set_xlabel('Number of Components')
    axs[0].set_ylim(0, 1)
    axs[0].set_xticks(r2x_v_components.index)
    axs[0].text(
        -0.25,
        0.9,
        'A',
        fontsize=14,
        fontweight='bold',
        transform=axs[0].transAxes
    )

    # R2X v. Scaling

    r2x_v_scaling.plot(ax=axs[1])
    axs[1].legend(
        ['Total', 'Cytokine', 'RNA']
    )
    axs[1].set_xscale("log")
    axs[1].set_ylabel('R2X')
    axs[1].set_xlabel('Variance Scaling\n(Cytokine/RNA)')
    axs[1].set_ylim(0, 1)
    axs[1].tick_params(axis='x', pad=-3)
    axs[1].text(
        -0.25,
        0.9,
        'B',
        fontsize=14,
        fontweight='bold',
        transform=axs[1].transAxes
    )

    # Accuracy v. Components

    axs[2].plot(acc_v_components.index, acc_v_components)
    axs[2].set_ylabel('Prediction Accuracy')
    axs[2].set_xlabel('Number of Components')
    axs[2].set_xticks(acc_v_components.index)
    axs[2].set_ylim([0.5, 0.75])
    axs[2].text(
        -0.25,
        0.9,
        'C',
        fontsize=14,
        fontweight='bold',
        transform=axs[2].transAxes
    )

    # Accuracy v. Scaling

    axs[3].semilogx(acc_v_scaling.index, acc_v_scaling, base=2)
    axs[3].set_ylabel('Prediction Accuracy')
    axs[3].set_xlabel('Variance Scaling\n(Cytokine/RNA)')
    axs[3].set_ylim([0.5, 0.75])
    axs[3].set_xticks(np.logspace(-16, 4, base=2, num=11))
    axs[3].tick_params(axis='x', pad=-3)
    axs[3].text(
        -0.25,
        0.9,
        'D',
        fontsize=14,
        fontweight='bold',
        transform=axs[3].transAxes
    )

    return fig


def makeFigure():
    r2x_v_components, acc_v_components, r2x_v_scaling, acc_v_scaling = get_r2x_results()

    fig = plot_results(
        r2x_v_components,
        r2x_v_scaling,
        acc_v_components,
        acc_v_scaling
    )

    return fig
