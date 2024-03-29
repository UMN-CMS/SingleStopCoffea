import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.collections import RegularPolyCollection, PatchCollection
from matplotlib.patches import Rectangle
import numpy as np
import hist.intervals as hinter
from . import static
from dataclasses import dataclass
from functools import partial, wraps
from analyzer.datasets import Style#, Dataset
from typing import Optional, Dict, Any, Union, Tuple
import itertools as it
import hist
import re
import importlib.resources as imp_res
from pathlib import Path


default_linewidth = 3


def loadStyles():
    HAS_IMPRES = hasattr(imp_res, "files")
    if HAS_IMPRES:
        style = imp_res.files(static) / "style.mplstyle"
        font_dirs = [imp_res.files(static) / "fonts"]
    else:
        from pkg_resources import resource_string, resource_listdir, resource_filename

        style = resource_filename("analyzer.plotting.static", "style.mplstyle")
        font_dirs = [resource_filename("analyzer.plotting.static", "fonts")]

    font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
    for font in font_files:
        font_manager.fontManager.addfont(font)
    plt.style.use(style)


def addPrelim(ax, pos="in", additional_text=None, color="black") -> mpl.axis.Axis:
    text = r"$\bf{CMS}\ \it{Preliminary}$"
    if additional_text:
        text += additional_text
    if pos == "in":
        ax.text(
            0.02,
            0.98,
            text,
            horizontalalignment="left",
            verticalalignment="top",
            transform=ax.transAxes,
            fontsize=20,
            color=color,
        )
    elif pos == "out":
        ax.text(
            0.02,
            1.0,
            text,
            horizontalalignment="left",
            verticalalignment="bottom",
            transform=ax.transAxes,
            fontsize=20,
            color=color,
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
    hist: Union[hist.Hist, tuple[np.ndarray]]
    title: Optional[str] = None
    dataset: Optional[Dataset] = None
    style: Optional[Dict[str, Any]] = None

    def ishist(self):
        return isinstance(self.hist, hist.Hist)

    def getBinCenters(self):
        if self.ishist():
            return tuple(x.centers for x in self.hist.axes)
        else:
            edges = self.hist[1]
            ret = edges[:-1] + np.diff(edges) / 2
            ret = np.atleast_2d(ret)
            return ret

    def getBinEdges(self):
        if self.ishist():
            return tuple(x.edges for x in self.hist.axes)
        else:
            edges = self.hist[1]
            edges = np.atleast_2d(edges)
            return edges

    def getValues(self):
        if self.ishist():
            return self.hist.values()
        else:
            return self.hist[0]

    def getUncertainty(self):
        if self.ishist():
            return np.sqrt(self.hist.variances())
        else:
            return self.hist[2]

    def getStyle(self):
        if self.style is not None:
            return self.style
        elif self.dataset:
            return self.dataset.style.toDict()
        else:
            return {}


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
    x = p.getBinCenters()[0]
    ed = p.getBinEdges()[0]
    y = p.getValues()
    var = p.getUncertainty()
    if yerr:
        ax.errorbar(
            x,
            y,
            yerr=var,
            fmt="none",
            linewidth=default_linewidth,
            **style,
            **kwargs,
        )
        ax.hlines(
            y,
            ed[1:],
            ed[:-1],
            label=p.title,
            linewidth=default_linewidth,
            **style,
        )
    else:
        ax.scatter(
            x,
            y,
            label=p.title,
            marker="+",
            linewidth=default_linewidth,
            **style,
            **kwargs,
        )
    return ax


@magicPlot
@autoSplit
def drawAs1DHist(ax, plot_object, plot_name = "", yerr=True, fill=True, orient="h", **kwargs):
    style = plot_object.getStyle()
    h = plot_object.hist
    x = plot_object.getBinCenters()[0]
    edges = plot_object.getBinEdges()[0]
    raw_vals = plot_object.getValues()
    vals = np.append(raw_vals, raw_vals[-1])
    errs = plot_object.getUncertainty()
    if yerr:
        if orient == "h":
            ax.errorbar(
                x,
                raw_vals,
                yerr=errs,
                fmt="none",
                linewidth=default_linewidth,
                **kwargs,
                **style,
            )
        else:
            ax.errorbar(
                raw_vals,
                x,
                xerr=errs,
                fmt="none",
                linewidth=default_linewidth,
                **style,
                **kwargs,
            )
    if fill:
        if orient == "h":
            ax.fill_between(
                edges,
                vals,
                step="post",
                label=plot_object.title,
                **style,
                **kwargs,
            )
        else:
            ax.fill_betweenx(
                edges,
                vals,
                step="post",
                label=plot_object.title,
                **style,
                **kwargs,
            )
    else:
        ax.stairs(
            raw_vals,
            edges,
            label=plot_object.title,
            linewidth=default_linewidth,
            orientation="vertical" if orient == "h" else "horizontal",
            **style,
            **kwargs,
        )

    return ax


@magicPlot
def drawRatio(
    ax, numerator, denominator, uncertainty_type="efficiency", hline_list=None, **kwargs
):
    hline_list = hline_list or []
    nh, dh = numerator.hist, denominator.hist
    an, ad = nh.axes[0], dh.axes[0]
    nv, dv = numerator.getValues(), denominator.getValues()
    ratio = np.divide(nv, dv, out=np.ones_like(nv), where=dv != 0)

    unc = hinter.ratio_uncertainty(
        numerator.getUncertainty(),
        denominator.getUncertainty(),
        uncertainty_type=uncertainty_type,
    )

    x = numerator.getBinCenters()[0]
    ax.errorbar(
        x,
        ratio,
        yerr=unc,
        marker="+",
        linestyle="none",
        **kwargs,
    )
    return ax


@magicPlot
def drawPull(ax, pred, obs, uncertainty_type="poisson", hline_list=None, **kwargs):
    hline_list = hline_list or []
    oh, ph = obs.hist, pred.hist
    ov, pv = obs.getValues(), pred.getValues()
    unc = pred.getUncertainty()
    ounc = obs.getUncertainty()
    real_unc = np.sqrt(unc**2 + ounc**2)
    real_unc = ounc
    pull = np.divide(
        ov - pv,
        real_unc,
        out=np.zeros_like(real_unc),
        where=real_unc != 0,
    )
    x = obs.getBinCenters()[0]
    ax.plot(
        x,
        pull,
        marker="o",
        markersize=2.5,
        linestyle="none",
        **kwargs,
    )
    for y in hline_list:
        ax.axhline(y, linewidth=1, linestyle="--", color="tab:grey")
    return ax


@magicPlot
def drawAs2DHist(ax, plot_object, divider=None, add_color_bar=True, **kwargs):
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
    ax.quadmesh = im[3]
    if divider is None:
        divider = make_axes_locatable(ax)
    if add_color_bar:
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(ax.quadmesh, cax=cax)
        cax.get_yaxis().set_offset_position("left")
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
    bottom_axes = getattr(ax, "bottom_axes", None)
    if bottom_axes:
        bottom_axes[-1].set_xlabel(lab)
    else:
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
        bottom = lim[0]
    else:
        bottom = lim[0] - delta * 0.05

    top = lim[1] + delta * top_pad

    ax.set_ylim(bottom, top)

    return ax


def addTitles2D(ax, hist):
    axes = hist.axes
    x_unit = getattr(axes[0], "unit", None)
    y_unit = getattr(axes[1], "unit", None)
    ax.set_xlabel(axes[0].label + (f" [{x_unit}]" if x_unit else ""))
    ax.set_ylabel(axes[1].label + (f" [{y_unit}]" if y_unit else ""))

    if hist.sum().value < 20:
        zlab = "Normalized Events"
    else:
        zlab = "Weighted Events"
    if x_unit and y_unit and x_unit == y_unit:
        zlab += f" / {x_unit or ''}$^2$"
    elif x_unit or y_unit:
        zlab += f" / {x_unit or ''} {y_unit or ''}"

    if hasattr(ax, "cax"):
        cax = ax.cax
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


@magicPlot
def addAxesToHist(
    ax,
    num_top=0,
    num_bottom=0,
    num_left=0,
    num_right=0,
    top_pad=0.2,
    right_pad=0.2,
    left_pad=0.2,
    bottom_pad=0.2,
    **plotopts,
):
    divider = make_axes_locatable(ax)
    ax.top_axes = []
    ax.bottom_axes = []
    ax.right_axes = []
    ax.left_axes = []
    for i in range(num_top):
        ax_histx = divider.append_axes("top", 1, pad=top_pad, sharex=ax)
        ax.top_axes.append(ax_histx)

    for i in range(num_bottom):
        ax_histx = divider.append_axes("bottom", 1, pad=bottom_pad, sharex=ax)
        ax.bottom_axes.append(ax_histx)

    for i in range(num_left):
        ax_histx = divider.append_axes("left", 1, pad=left_pad, sharey=ax)
        ax.left_axes.append(ax_histx)

    for i in range(num_right):
        ax_histx = divider.append_axes("right", 1, pad=right_pad, sharey=ax)
        ax.right_axes.append(ax_histx)
    return ax


@magicPlot
def drawAs2DExtended(
    ax,
    plot_object_2d,
    top_stack=None,
    right_stack=None,
    color_bar=False,
    top_pad=0.2,
    right_pad=0.2,
    **plotopts,
):
    divider = make_axes_locatable(ax)
    ax = drawAs2DHist(
        ax, plot_object_2d, divider=divider, add_color_bar=False, **plotopts
    )

    ax.top_axes = []
    ax.right_axes = []
    for plot_object in top_stack or []:
        ax_histx = divider.append_axes("top", 1, pad=top_pad, sharex=ax)
        drawAs1DHist(ax_histx, plot_object)
        ax_histx.xaxis.set_tick_params(labelbottom=False)
        ax.top_axes.append(ax_histx)

    for plot_object in right_stack or []:
        ax_histy = divider.append_axes("right", 1, pad=right_pad, sharey=ax)
        drawAs1DHist(ax_histy, plot_object, orient="v")
        ax_histy.yaxis.set_tick_params(labelleft=False)
        ax.right_axes.append(ax_histy)

    if color_bar:
        cax = divider.append_axes("left", size="5%", pad=1.5)
        cbar = plt.colorbar(ax.quadmesh, cax=cax, orientation="vertical")
        ax.cax = cax
    return ax


@dataclass
class AnnotRectangle:
    x: float
    y: float
    w: float
    h: float
    text: str
    value: float


def drawRectangles(ax, rects):
    ymax = max(r.y + r.h for r in rects)
    xmax = max(r.x + r.w for r in rects)
    ymin = min(r.y for r in rects)
    xmin = min(r.x for r in rects)
    ax.set_xlim((xmin, xmax))
    ax.set_ylim((ymin, ymax))
    patches = [
        Rectangle(
            (r.x, r.y),
            r.w,
            r.h,
        )
        for r in rects
    ]
    col = PatchCollection(patches)
    col.set_array([r.value for r in rects])
    ax.add_collection(col)
    ax.autoscale_view()
    for r in rects:
        ax.annotate(
            r.text,
            (r.x + r.w / 2, r.y + r.h / 2),
            color="w",
            weight="bold",
            fontsize=6,
            ha="center",
            va="center",
        )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    cbar = plt.colorbar(col, cax=cax)
    cax.get_yaxis().set_offset_position("left")
    ax.cax = cax
    return ax
