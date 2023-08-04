"""
Creates Figure 2 -- CMTF Plotting
"""
import numpy as np
import pandas as pd

from .common import getSetup
from ..cmtf import perform_CMTF, perform_pca
from ..dataImport import form_tensor
from ..predict import run_model
from tensorpack import calcR2X


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
    components = 10

    r2x_v_components = pd.DataFrame(
        columns=['CMTF', 'PCA'],
        index=np.arange(1, components + 1),
        dtype=float
    )
    acc_v_components = pd.DataFrame(
        columns=['CMTF', 'PCA'],
        index=np.arange(1, components + 1).tolist(),
        dtype=float
    )
    for n_components in r2x_v_components.index:
        print(f"Starting decomposition with {n_components} components.")
        t_fac = perform_CMTF(tensor, matrix, r=n_components)
        pca_components, pca_var = perform_pca(
            tensor,
            matrix,
            r=n_components
        )
        r2x_v_components.loc[n_components, 'CMTF'] = t_fac.R2X
        r2x_v_components.loc[n_components, 'PCA'] = pca_var
        acc_v_components.loc[n_components, 'CMTF'] = \
            run_model(t_fac.factors[0], labels)[0]
        acc_v_components.loc[n_components, 'PCA'] = \
            run_model(pca_components, labels)[0]

    # R2X v. Scaling
    scalingV = np.logspace(-10, 10, base=2, num=21)
    r2x_v_scaling = pd.DataFrame(
        index=scalingV,
        columns=["Total", "Tensor", "Matrix"]
    )
    acc_v_scaling = pd.DataFrame(
        columns=['CMTF', 'PCA'],
        index=scalingV.tolist(),
        dtype=float
    )
    for scaling in r2x_v_scaling.index:
        tensor, matrix, _ = form_tensor(scaling)
        t_fac = perform_CMTF(tensor, matrix)
        pca_components, _ = perform_pca(tensor, matrix)
        r2x_v_scaling.loc[scaling, "Total"] = t_fac.R2X
        r2x_v_scaling.loc[scaling, "Tensor"] = calcR2X(t_fac, tIn=tensor)
        r2x_v_scaling.loc[scaling, "Matrix"] = calcR2X(t_fac, mIn=matrix)
        acc_v_scaling.loc[scaling, 'CMTF'] = \
            run_model(t_fac.factors[0], labels)[0]
        acc_v_scaling.loc[scaling, 'PCA'] = \
            run_model(pca_components, labels)[0]

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
    axs[0].plot(r2x_v_components.index, r2x_v_components.loc[:, 'CMTF'])
    axs[0].plot(r2x_v_components.index, r2x_v_components.loc[:, 'PCA'])
    axs[0].legend(['CMTF', 'PCA'])
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

    axs[1].semilogx(r2x_v_scaling.index, r2x_v_scaling.loc[:, 'Total'], base=2)
    axs[1].semilogx(r2x_v_scaling.index, r2x_v_scaling.loc[:, 'Tensor'], base=2)
    axs[1].semilogx(r2x_v_scaling.index, r2x_v_scaling.loc[:, 'Matrix'], base=2)
    axs[1].legend(
        ['Total', 'Cytokine', 'RNA']
    )
    axs[1].set_ylabel('R2X')
    axs[1].set_xlabel('Variance Scaling\n(Cytokine/RNA)')
    axs[1].set_ylim(0, 1)
    axs[1].set_xticks(np.logspace(-10, 10, base=2, num=11))
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

    axs[2].plot(acc_v_components.index, acc_v_components.loc[:, 'CMTF'])
    axs[2].plot(acc_v_components.index, acc_v_components.loc[:, 'PCA'])
    axs[2].legend(['CMTF', 'PCA'])
    axs[2].set_ylabel('Prediction Accuracy')
    axs[2].set_xlabel('Number of Components')
    axs[2].set_xticks(acc_v_components.index)
    axs[2].set_ylim([0.45, 0.75])
    axs[2].text(
        -0.25,
        0.9,
        'C',
        fontsize=14,
        fontweight='bold',
        transform=axs[2].transAxes
    )

    # Accuracy v. Scaling

    axs[3].semilogx(acc_v_scaling.index, acc_v_scaling.loc[:, 'CMTF'], base=2)
    axs[3].semilogx(acc_v_scaling.index, acc_v_scaling.loc[:, 'PCA'], base=2)
    axs[3].legend(['CMTF', 'PCA'])
    axs[3].set_ylabel('Prediction Accuracy')
    axs[3].set_xlabel('Variance Scaling\n(Cytokine/RNA)')
    axs[3].set_yticks(np.arange(0.5, 0.75, 0.05))
    axs[3].set_ylim([0.5, 0.7])
    axs[3].set_xticks(np.logspace(-10, 10, base=2, num=11))
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
