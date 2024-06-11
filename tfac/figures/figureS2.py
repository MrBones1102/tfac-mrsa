"""
This creates Figure S2 - Full Cytokine plots
"""
import matplotlib
from matplotlib.patches import Patch
import numpy as np

from tfac.figures.common import getSetup
from tfac.dataImport import import_cytokines

COLOR_CYCLE = matplotlib.rcParams['axes.prop_cycle'].by_key()['color']


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    fig_size = (8, 4)
    layout = {
        'ncols': 1,
        'nrows': 1
    }
    axs, f, _ = getSetup(
        fig_size,
        layout
    )
    ax = axs[0]

    plasma, serum = import_cytokines()
    ax.boxplot(
        serum.T,
        positions=np.arange(0, serum.shape[0] * 3, 3),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[0],
        },
        flierprops={
            'markersize': 4,
            'markerfacecolor': COLOR_CYCLE[0]
        }
    )
    ax.boxplot(
        plasma.T,
        positions=np.arange(1, plasma.shape[0] * 3, 3),
        sym='d',
        patch_artist=True,
        boxprops={
            'facecolor': COLOR_CYCLE[1],
        },
        flierprops={
            'markersize': 4,
            'markerfacecolor': COLOR_CYCLE[1]
        }
    )

    legend_patches = [
        Patch(color=COLOR_CYCLE[0]),
        Patch(color=COLOR_CYCLE[1])
    ]
    ax.legend(legend_patches, ['Serum', 'Plasma'])

    ax.set_xlim([-2, serum.shape[0] * 3])
    ax.set_xticks(np.arange(0.5, serum.shape[0] * 3, 3))
    ax.set_xticklabels(
        plasma.index,
        rotation=90,
        va='top'
    )
    ax.set_xlabel('Cytokine')
    ax.set_ylabel('Normalized Cytokine Level')
    ax.set_title('Cytokine Level vs. Source')

    return f
