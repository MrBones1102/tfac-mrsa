"""
This creates Figure S1 - Full Cytokine plots
"""
from matplotlib.patches import PathPatch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
import seaborn as sns

from tfac.figures.common import getSetup
from tfac.dataImport import form_tensor, import_cytokines


def adjust_box_widths(g, fac):
    """
    Adjust the widths of a seaborn-generated boxplot.

    Function adapted from: https://github.com/mwaskom/seaborn/issues/1076
    """
    # iterating through Axes instances
    for ax in g.axes:

        # iterating through axes artists:
        for c in ax.get_children():

            # searching for PathPatches
            if isinstance(c, PathPatch):
                # getting current width of box:
                p = c.get_path()
                verts = p.vertices
                verts_sub = verts[:-1]
                xmin = np.min(verts_sub[:, 0])
                xmax = np.max(verts_sub[:, 0])
                xmid = 0.5*(xmin+xmax)
                xhalf = 0.5*(xmax - xmin)

                # setting new width of box
                xmin_new = xmid-fac*xhalf
                xmax_new = xmid+fac*xhalf
                verts_sub[verts_sub[:, 0] == xmin, 0] = xmin_new
                verts_sub[verts_sub[:, 0] == xmax, 0] = xmax_new

                # setting new width of median line
                for l in ax.lines:
                    if np.all(l.get_xdata() == [xmin, xmax]):
                        l.set_xdata([xmin_new, xmax_new])


def fig_S1_setup():
    tensor, _, patInfo = form_tensor()
    plasma, _ = import_cytokines()
    cytokines = plasma.index
    n_cytokines = len(cytokines)

    tensor = tensor.T
    patInfo = patInfo.T
    serum_slice = tensor[0, :, :]
    plasma_slice = tensor[1, :, :]

    test = pd.concat([pd.DataFrame(serum_slice), pd.DataFrame(plasma_slice)])
    test = test.dropna(axis=1)
    pears = []
    for i in range(n_cytokines):
        pears.append([cytokines[i], pearsonr(test.iloc[i, :].to_numpy(dtype=float), test.iloc[i + n_cytokines, :].to_numpy(dtype=float))[0]])
    pears = pd.DataFrame(pears).sort_values(1)

    return pears, serum_slice, plasma_slice, cytokines, patInfo


def cytokine_boxplot(cyto_slice, cytokines, patInfo, axx):
    ser = pd.DataFrame(cyto_slice, index=cytokines, columns=patInfo.columns).T
    patInfo = patInfo.T["status"]
    ser = ser.join(patInfo).dropna().reset_index().melt(id_vars=["sid", "status"])
    b = sns.boxplot(
        data=ser,
        x="variable",
        y="value",
        hue="status",
        linewidth=1,
        ax=axx
    )
    b.set_xticklabels(b.get_xticklabels(), rotation=30, ha="right")
    b.set_xlabel("Cytokine")
    b.set_ylabel("Normalized cytokine level")


def makeFigure():
    """ Get a list of the axis objects and create a figure. """
    # Get list of axis objects
    fig_size = (8, 8)
    layout = {
        'ncols': 1,
        'nrows': 3
    }
    ax, f, _ = getSetup(
        fig_size,
        layout
    )
    pears, serum_slice, plasma_slice, cytokines, patInfo = fig_S1_setup()
    a = sns.pointplot(data=pears, x=0, y=1, join=False, ax=ax[0])
    a.set_xticklabels(a.get_xticklabels(), rotation=30, ha="right")
    a.set_xlabel("Cytokine")
    a.set_ylabel("Pearson's correlation")
    a.set_title("Serum-Plasma Cytokine Level Correlation")

    cytokine_boxplot(serum_slice, cytokines, patInfo, ax[1])
    ax[1].set_title("Normalized Serum Cytokine Level by Outcome")

    cytokine_boxplot(plasma_slice, cytokines, patInfo, ax[2])
    ax[2].set_title("Normalized Plasma Cytokine Level by Outcome")

    # adjust_box_widths(f, 0.75)

    return f
