import numpy as np
import csv
import decimal
from collections import defaultdict

import matplotlib as mpl
import matplotlib.pyplot as plt
import mplhep
from analyzer.postprocessing.style import Styler

from ..grouping import doFormatting
from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .mplstyles import loadStyles
from .utils import addAxesToHist, fixBadLabels, saveFig
from pathlib import Path

def getRatioAndUnc(num, den, uncertainty_type="poisson-ratio"):
    import hist.intervals as hinter

    with np.errstate(divide="ignore", invalid="ignore"):
        ratios = num / den
        unc = hinter.ratio_uncertainty(
            num=num, denom=den, uncertainty_type=uncertainty_type
        )
    return ratios, unc


def plotOne(
    packaged_hists,
    group_params,
    output_path,
    style_set,
    scale="linear",
    normalize=False,
    plot_configuration=None,
    stacked_hists=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    fig, ax = plt.subplots()
    for packaged_hist in packaged_hists:
        title = packaged_hist.title
        h = packaged_hist.histogram
        fixBadLabels(h)
        style = styler.getStyle(packaged_hist.sector_parameters)

        new_n = np.where(h.values()!=0, h.values(), np.nan*h.values())
        new_variance = np.where(h.variances()!=0, h.variances(), np.nan*h.variances())
        h[...] = np.stack([new_n, new_variance], axis=-1)
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            yerr=style.yerr,
            flow="none",
            **style.get(),
        )

    if stacked_hists:
        stacked_hists = sorted(stacked_hists, key=lambda x: x.histogram.sum().value)
        style_kwargs = defaultdict(list)
        hists = []
        titles = []
        for x in stacked_hists:
            hists.append(x.histogram)
            titles.append(x.title)
            style = styler.getStyle(x.sector_parameters)
            for k, v in style.get().items():
                style_kwargs[k].append(v)

        style_kwargs["histtype"] = style_kwargs["histtype"][0]

        mplhep.histplot(
            hists,
            ax=ax,
            stack=True,
            **style_kwargs,
            label=titles,  # sort="yield"
        )

    labelAxis(ax, "x", h.axes, label=plot_configuration.x_label)

    if normalize:
        y_label = "Normalized Events"
    else:
        y_label = plot_configuration.y_label 

    labelAxis(ax, "y", h.axes, label=y_label)
    addCMSBits(
        ax,
        [x.sector_parameters for x in packaged_hists],
        plot_configuration=pc,
    )
    if style.legend:
        legend_kwargs = {}
        if style.legend_font:
            legend_kwargs["fontsize"] = style.legend_font
        ax.legend(loc="upper right", **legend_kwargs)
        mplhep.sort_legend(ax=ax)
    ax.set_yscale(scale)
    if style.y_min:
        ax.set_ylim(bottom=style.y_min)
    else:
        mplhep.ylow(ax)
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def makeStrHist(data, ax_name=None):
    import hist

    ax = hist.axis.StrCategory([x[0] for x in data], name=ax_name)
    h = hist.Hist(ax, storage="double")
    h[:] = np.array([x[1] for x in data])
    return h


def __plotStrCatOne(
    getter,
    group_params,
    sectors,
    output_path,
    style_set,
    ax_name=None,
    normalize=False,
    scale="linear",
    init_normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    loadStyles()
    mpl.use("Agg")

    fig, ax = plt.subplots(layout="constrained")
    for sector in sectors:
        if init_normalize:
            p = sector.sector_params
            style = styler.getStyle(p)
            h = makeStrHist(getter(sector), ax_name=ax_name)
            #initial = h[2]
            #h = h/initial
            #h = h[2:]
            h.plot1d(
                ax=ax,
                label=sector.sector_params.dataset.title,
                **style.get(),
            )
        else:
            p = sector.sector_params
            style = styler.getStyle(p)
            h = makeStrHist(getter(sector), ax_name=ax_name)
            h.plot1d(
                ax=ax,
                label=sector.sector_params.dataset.title,
                density=normalize,
                **style.get(),
            )
        #initial = h[0]
        #for index, x in enumerate(ax.get_xticks()):
        #    ax.text(x,h[index],h[index]/initial,horizontalalignment='center', verticalalignment='bottom', fontsize=10)
    ax.legend()
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    ax.tick_params(axis="x", rotation=90)
    addCMSBits(
        ax,
        [x.sector_params for x in sectors],
        plot_configuration=pc,
    )
    ax.set_yscale(scale)
    # mplhep.yscale_legend(ax, soft_fail=True)
    ax.legend(loc="upper right")
    mplhep.sort_legend(ax=ax)
    # mplhep.yscale_legend(ax, soft_fail=True)
    saveFig(fig, output_path, extension=pc.image_type)
    #output = Path(output_path).parent.parent.parent/"eff_2btag.csv"
#    eff = h[-1]/initial
#    with open(output, "a") as f:
#        writer = csv.writer(f, delimiter=",")
#        writer.writerow([sector.sector_params.dataset.title,eff])
#        
#
    plt.close(fig)


def __plotStrCatAsTable(
    getter,
    sectors,
    group_params,
    output_name,
    style_set,
    ax_name=None,
    normalize=False,
    plot_configuration=None,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    loadStyles()
    mpl.use("Agg")

    rep_data = getter(sectors[0])
    col_labels = [x[0] for x in rep_data]
    rows = []
    row_labels = []

    figsize = (len(rep_data) * 0.3, len(sectors) * 0.3)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    fig.patch.set_visible(False)
    ax.axis("off")
    ax.axis("tight")
    for sector in sectors:
        p = sector.sector_params
        styler.getStyle(p)
        data = getter(sector)
        row_labels.append(sector.sector_params.dataset.title)
        rows.append([x[1] for x in data])

    table = ax.table(
        cellText=rows,
        rowLabels=row_labels,
        colLabels=col_labels,
    )

    o = doFormatting(output_name, group_params, histogram_name=(ax_name or ""))
    saveFig(fig, o, extension=pc.image_type)
    plt.close(fig)


def plotStrCat(plot_type, *args, table_mode=False, **kwargs):
    def makeGetter(n):
        def inner(sec):
            return getattr(sec.result.selection_flow, n)

        return inner

    if table_mode:
        f = __plotStrCatAsTable
    else:
        f = __plotStrCatOne

    f(makeGetter(plot_type), *args, ax_name=plot_type, **kwargs)


def plotRatio(
    denominator,
    numerators,
    output_path,
    style_set,
    normalize=False,
    ratio_ylim=(0, 2),
    ratio_type="poisson",
    scale="linear",
    plot_configuration=None,
    ratio_hlines=(1.0,),
    ratio_height=1.5,
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)

    fig, ax = plt.subplots(layout="constrained")
    ratio_ax = addAxesToHist(ax, size=ratio_height, pad=0.3)

    den_hist = denominator.histogram

    fixBadLabels(den_hist)

    style = denominator.style or styler.getStyle(denominator.sector_parameters)

    den_hist.plot1d(
        ax=ax,
        label=denominator.title+" 1% mistag",
        density=normalize,
        yerr=True,
        color='green',
        alpha=0.4,
        linewidth=1.5,
        histtype='fill',
    )

    den_hist_5 = den_hist*(0.0025/0.0001)
    den_hist_5.plot1d(
        ax=ax,
        label=denominator.title+" 5% mistag",
        density=normalize,
        yerr=True,
        color="limegreen",
        alpha=0.3,
        linewidth=1.5,
        histtype='fill',
        zorder=0,
    )
    #ax.set_ylim(0.1, 10.5**5)

    x_values = den_hist.axes[0].centers
    left_edge = den_hist.axes.edges[0][0]
    right_edge = den_hist.axes.edges[-1][-1]

    all_ratios, all_uncertainties = [], []

    for num in numerators:
        title = num.title
        h = num.histogram
        fixBadLabels(h)
        num.sector_parameters
        s = num.style or styler.getStyle(num.sector_parameters)

        n, d = h.values(), den_hist.values()

        ratio, unc = getRatioAndUnc(n, d, uncertainty_type=ratio_type)
        if normalize:
            with np.errstate(divide="ignore", invalid="ignore"):
                ratio = (n / np.sum(n)) / (d / np.sum(d))
        all_ratios.append(ratio)
        all_uncertainties.append(unc)
        #zero mask
        new_n = np.where(n!=0, n, np.nan*n)
        new_variance = np.where(n!=0, h.variances(), np.nan*h.variances())
        h[...] = np.stack([new_n, new_variance], axis=-1)
        h.plot1d(
            ax=ax,
            label=title,
            density=normalize,
            yerr=True,
            **s.get(),
        )

        ratio[ratio == 0] = np.nan
        ratio[np.isinf(ratio)] = np.nan
        all_opts = {**s.get("errorbar"), **dict(linestyle="none")}
        all_opts.pop("histtype")

        #current_color = ax.lines[-1].get_color()
        #ratio_ax.scatter(
        #    x_values,
        #    ratio,
        #    label=title,
        #    color=current_color,
        #)
        #replace with significance
        sig = np.sqrt(2*((n+d)*np.log(1+(n/d)) - n)) 
        d5 = den_hist_5.values()
        sig_5 = np.sqrt(2*((n+d5)*np.log(1+(n/(d5))) - n)) 
        summed_sig = decimal.Decimal(np.sqrt(np.nansum(sig**2)))
        summed_sig5 = decimal.Decimal(np.sqrt(np.nansum(sig_5**2)))
        current_color = ax.lines[-1].get_color()
 
        ratio_ax.scatter(
            x_values,
            sig,
            label=f'{summed_sig:.2g}$\sigma$',
            color=current_color,
        )
        ratio_ax.scatter(
            x_values,
            sig_5,
            label=f'{summed_sig5:.2g}$\sigma$',
            marker='x',
            color=current_color,
        )
        

    for l in ratio_hlines:
        ratio_ax.axhline(l, color="black", linestyle="dashed", linewidth=1.0)

    ratio_ax.set_xlim(left_edge, right_edge)
    ratio_ax.legend(title="Total Significance: 1% mistag vs 5% mistag", ncols=3, fontsize=16, title_fontsize=16)
    ratio_ax.set_ylim(bottom=ratio_ylim[0], top=ratio_ylim[1])
    #if normalize:
    #    y_label = "Normalized Events"
    #else:
    #    y_label = None

    #labelAxis(ax, "y", den_hist.axes, label=y_label)
    ax.legend(loc="upper right", fontsize=18)
    ax.set_xlabel(None)
    ax.set_ylabel("Events", fontsize=24)
    labelAxis(ratio_ax, "x", den_hist.axes)
    addCMSBits(
        ax,
        [denominator.sector_parameters, *(x.sector_parameters for x in numerators)],
        plot_configuration=pc,
    )


    ratio_ax.set_ylabel("Significance", loc="center", fontsize=18)
    #ratio_ax.set_ylabel("Ratio", loc="center", fontsize=15)
    #ax.set_xlim(250, 3500)
    ax.tick_params(axis="x", which="both", labelbottom=False)
    mplhep.sort_legend(ax=ax)
    ax.set_yscale(scale)
    fig.tight_layout()
    saveFig(fig, output_path, extension=pc.image_type)
