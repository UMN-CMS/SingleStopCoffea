import itertools as it

import matplotlib.pyplot as plt

from .annotations import addEra, addPrelim
from .plots_1d import addTitles1D, drawAs1DHist, drawPull, drawRatio
from .plots_2d import addTitles2D, drawAs2DHist
from .utils import addAxesToHist


def plotPulls(plotobj_pred, plotobj_obs, coupling, lumi):
    fig, ax = plt.subplots()

    hopo = plotobj_obs
    hppo = plotobj_pred
    ax = drawAs1DHist(hopo, yerr=True, fill=False)

    drawAs1DHist(ax, hppo, yerr=True, fill=False)
    addAxesToHist(ax, num_bottom=1, bottom_pad=0)

    ab = ax.bottom_axes[0]
    drawPull(ab, hppo, hopo)
    ab.set_ylabel(r"$\frac{pred - obs}{\sigma_{pred}}$")
    addEra(ax, lumi or 59.8)
    addPrelim(ax, additional_text=f"\n$\\lambda_{{{coupling}}}''$ ")
    addTitles1D(ax, hopo, top_pad=0.2)
    fig.tight_layout()
    return fig


def plotRatio(plotobj_pred, plotobj_obs, coupling, lumi, no_hists=False, ax=None):

    hopo = plotobj_obs
    hppo = plotobj_pred

    if not no_hists:
        fig, ax = plt.subplots()

        drawAs1DHist(ax, hopo, yerr=True, fill=False)
        drawAs1DHist(ax, hppo, yerr=True, fill=False)

    addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    ab = ax.bottom_axes[0]
    drawRatio(ab, hppo, hopo)

    ab.set_ylabel("Ratio")
    ab.set_ylim(0,2)
    addEra(ax, lumi)
    addPrelim(ax, additional_text=f"\n$\\lambda_{{{coupling}}}''$ ")
    addTitles1D(ax, hopo, top_pad=0.2)
    
    if no_hists:
        return ax
    else:
        fig.tight_layout()
        return fig

def plot1D(
    signal_plobjs,
    background_plobjs,
    lumi,
    coupling,
    sig_style="hist",
    scale="log",
    xlabel_override=None,
    add_label=None,
    top_pad=0.4,
    ratio=False,
    un_sig_objs = None,
):
    fig, ax = plt.subplots()

    for o in background_plobjs:
        drawAs1DHist(ax, o, yerr=False)
    for o in signal_plobjs:
        # drawAs1DHist(ax, o, yerr=False)
        if sig_style == "scatter":
            drawAsScatter(ax, o, yerr=True)
        elif sig_style == "hist":
            drawAs1DHist(ax, o, yerr=True, fill=False)
    if ratio and un_sig_objs is not None:
        plotRatio(un_sig_objs[0],un_sig_objs[1],coupling,lumi,no_hists=True,ax=ax)
    elif ratio:
        plotRatio(signal_plobjs[0],signal_plobjs[1],coupling,lumi,no_hists=True,ax=ax)
    ax.set_yscale(scale)
    addEra(ax, lumi)
    addPrelim(
        ax,
        additional_text=f"\n$\\lambda_{{{coupling}}}''$ Selection\n"
        + (add_label or ""),
    )
    hc = next(it.chain(signal_plobjs, background_plobjs))
    addTitles1D(ax, hc, top_pad=top_pad)
    handles, labels = ax.get_legend_handles_labels()
    labels, handles = zip(*reversed(sorted(zip(labels, handles), key=lambda t: t[0])))
    extra_legend_args = {}
    if len(labels) > 5:
        extra_legend_args["prop"] = {"size": 10}
    ax.legend(handles, labels, **extra_legend_args)
    if xlabel_override:
        ax.set_xlabel(xlabel_override)
    fig.tight_layout()
    return fig


def plot2D(
    plot_obj,
    lumi,
    coupling,
    sig_style="hist",
    scale="log",
    add_label=None,
):
    fig, ax = plt.subplots()

    drawAs2DHist(ax, plot_obj)
    addEra(ax, lumi)
    pos = "in"
    addPrelim(
        ax,
        additional_text=f"\n$\\lambda_{{{coupling}}}''$ Selection\n"
        + (f"{add_label}," if add_label else "")
        + f"{plot_obj.title}",
        pos=pos,
        color="white",
    )
    addTitles2D(ax, plot_obj)
    return fig
