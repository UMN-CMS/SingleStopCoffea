import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager

from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import importlib.resources as imp_res
from . import static
from dataclasses import dataclass
from functools import partial, wraps
from analyzer.datasets import Style, Dataset
from typing import Optional
import itertools as it
import hist
import re


default_linewidth = 3


def loadStyles():
    style = imp_res.files(static) / "style.mplstyle"

    font_dirs = [imp_res.files(static) / "fonts"]
    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
    plt.style.use(style)


def addPrelim(ax, pos="in") -> mpl.axis.Axis:
    if pos == "in":
        ax.text(
            0.02,
            0.98,
            r"$\bf{CMS}\ \it{Preliminary}$",
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=20,
        )
    elif pos == "out":
        ax.text(
            0.02,
            1.0,
            r"$\bf{CMS}\ \it{Preliminary}$",
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=20,
        )
    return ax


def addEra(ax, lumi) -> mpl.axis.Axis:
    ax.text(
        1,
        1,
        f"${lumi}\\, \\mathrm{{fb}}^{{-1}}$ (13 TeV) ",
        horizontalalignment="right",
        verticalalignment="bottom",
        transform=ax.transAxes,
        fontsize=20,
    )
    return ax


def histRank(hist):
    return len(hist.axes)


@dataclass
class PlotObject:
    hist: hist.Hist
    title: Optional[str]
    dataset: Optional[Dataset]

    def getStyle(self):
        if self.dataset:
            return self.dataset.style


def autoSplit(func):
    @wraps(func)
    def inner(ax, hist, cat_axis=None, cat_filter=None, manager=None, **kwargs):

        if cat_axis is None:
            if isinstance(hist, PlotObject):
                return func(ax, hist, **kwargs)
            else:
                return func(ax, PlotObject(hist), **kwargs)
        else:
            other_axes = [a for a in hist.axes if a.name != cat_axis]
            titles = [a.label for a in other_axes]
            ax_title = hist.name
            to_plot = [
                PlotObject(hist[{cat_axis: n}], manager[n].getTitle(), manager[n])
                for n in hist.axes[cat_axis]
                if cat_filter is None or (re.search(cat_filter, n))
            ]
            to_plot = reversed(sorted(to_plot, key=lambda x: x.hist.sum().value))
            for p in to_plot:
                ax = func(ax, p, **kwargs)
            ax.set_xlabel(titles[0])
            if len(other_axes) > 1:
                ax.set_ylabel(titles[1])
            else:
                ax.set_ylabel("Events")
            ax.legend()
            return ax

    return inner


def magicPlot(func):
    @wraps(func)
    def inner(first, *args, **kwargs):
        if isinstance(first, plt.Axes):
            return func(first, *args, **kwargs)
        else:
            fig, ax = plt.subplots()
            return fig, func(ax, first, *args, **kwargs)

    return inner


@magicPlot
@autoSplit
def drawAsScatter(ax, p, yerr=True, **kwargs):
    style = p.getStyle()
    hist = p.hist
    x = hist.axes.centers[0]
    ed = hist.axes.edges[0]
    y = hist.values()
    var = np.sqrt(hist.variances())
    if yerr:
        ax.errorbar(
            x,
            y,
            yerr=var,
            fmt="none",
            linewidth=default_linewidth,
            **style.toDict(),
            **kwargs,
        )
        ax.hlines(
            y,
            ed[1:],
            ed[:-1],
            label=p.title,
            linewidth=default_linewidth,
            **style.toDict(),
        )
    else:
        ax.scatter(
            x,
            y,
            label=p.title,
            marker="+",
            linewidth=default_linewidth,
            **style.toDict(),
            **kwargs,
        )
    return ax


@magicPlot
@autoSplit
def drawAs1DHist(ax, plot_object, yerr=True, fill=True, **kwargs):
    style = plot_object.getStyle()
    h = plot_object.hist
    a = h.axes[0]
    x = a.centers
    edges = a.edges
    raw_vals = h.values()
    vals = np.append(raw_vals, raw_vals[-1])
    errs = np.sqrt(h.variances())

    if yerr:
        ax.errorbar(
            x,
            raw_vals,
            yerr=errs,
            fmt="none",
            linewidth=default_linewidth,
            **style.toDict(),
            **kwargs,
        )
    if fill:
        ax.fill_between(
            edges,
            vals,
            step="post",
            label=plot_object.title,
            **style.toDict(),
            **kwargs,
        )
    else:
        ax.stairs(
            raw_vals,
            edges,
            label=plot_object.title,
            linewidth=default_linewidth,
            **style.toDict(),
            **kwargs,
        )

    return ax


@magicPlot
def drawAs2DHist(ax, plot_object, **kwargs):
    h = plot_object.hist
    a1 = h.axes[0]
    a2 = h.axes[1]
    vals, e1, e2 = h.to_numpy()
    ex = a1.centers
    ey = a2.centers
    vx, vy = np.meshgrid(ex, ey)
    x = vx.ravel()
    y = vy.ravel()
    w = vals.T.ravel()
    im = ax.hist2d(x, y, bins=[e1, e2], weights=w, **kwargs)
    ax.set_xlabel(a1.label)
    ax.set_ylabel(a2.label)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(im[3], cax=cax)
    ax.cax = cax
    return ax


def set_xmargin(ax, left=0.0, right=0.3):
    ax.set_margin(0)
    ax.autoscale_view()
    lim = ax.get_xlim()
    delta = np.diff(lim)
    left = lim[0] - delta * left
    right = lim[1] + delta * right
    ax.set_xlim(left, right)


def addTitles1D(ax, hist, exclude=None, top_pad=0.3):
    exclude = exclude or {
        "dataset",
    }
    axes = [x for x in hist.axes if x.name not in exclude]
    unit = getattr(axes[0], "unit", None)
    lab = axes[0].label
    if unit:
        lab += f" [{unit}]"
    ax.set_xlabel(lab)
    has_var = hist.variances()
    if hist.sum().value < 20:
        ylab = "Normalized Events"
    elif has_var is None:
        ylab = "Events"
    else:
        ylab = "Weighted Events"
    if unit:
        ylab += f" / {unit}"
    ax.set_ylabel(ylab)

    sc = ax.get_yscale()
    ax.set_ymargin(0)
    ax.autoscale_view()
    lim = ax.get_ylim()
    delta = np.diff(lim)
    if sc == "log":
        top_pad = 10 ** (1 + top_pad)
        bottom = max(1, lim[0] - delta * 0.05)
    else:
        bottom = lim[0] - delta * 0.05

    top = lim[1] + delta * top_pad

    ax.set_ylim(bottom, top)

    return ax


def addTitles2D(ax, hist):
    axes = hist.axes
    cax = ax.cax
    x_unit = getattr(axes[0], "unit", None)
    y_unit = getattr(axes[1], "unit", None)
    ax.set_xlabel(axes[0].label + f" [{x_unit}]" if x_unit else "")
    ax.set_ylabel(axes[1].label + f" [{y_unit}]" if x_unit else "")
    if hist.sum().value < 20:
        zlab = "Normalized Events"
    else:
        zlab = "Weighted Events"
    if x_unit and y_unit and x_unit == y_unit:
            zlab += f" / {x_unit or ''}$^2$"
    elif x_unit or y_unit:
        zlab += f" / {x_unit or ''} {y_unit or ''}"

    cax.set_ylabel(zlab)
    return ax


def getNormalized(hist, dataset_axis=None):
    ret = hist.copy(deep=True)
    if dataset_axis is None:
        val, var = ret.values(flow=True), ret.variances(flow=True)
        s = ret.sum(flow=True).value
        ret[...] = np.stack([val / s, var / s**2], axis=-1)
    else:
        for x in ret.axes[dataset_axis]:
            this = ret[{dataset_axis: x}]
            val, var = this.values(flow=True), this.variances(flow=True)
            s = this.sum(flow=True).value
            ret[{dataset_axis: x}] = np.stack([val / s, var / s**2], axis=-1)
    return ret
