import pickle as pkl
import uproot
import matplotlib.pyplot as plt
import hist
import math
import itertools as it
import re
import numpy as np
from pathlib import Path
import json
import sys
import argparse
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    DotProduct,
    ConstantKernel,
    WhiteKernel,
    RBF,
    Kernel,
    Matern,
    Hyperparameter,
)
import re
from analyzer.plotting.core_plots import (
    loadStyles,
    PlotObject,
    drawPull,
    drawAs1DHist,
    addAxesToHist,
    addTitles1D,
    addPrelim,
    addEra,
    drawAsScatter,
)
from analyzer.plotting.simple_plot import Plotter

loadStyles()


def fitGp(histogram, length_scale=None, window=None):
    vals, points, varia = (
        histogram.values(),
        histogram.axes[0].centers.reshape(-1, 1),
        np.sqrt(histogram.variances()),
    )
    if window:
        mask = np.ravel((points < window[0]) | (points > window[1]))
        vals,points,varia=vals[mask],points[mask],varia[mask]
    if length_scale: 
        # kernel = ConstantKernel(10.0, constant_value_bounds=(0.1, 1e13)) * Matern(
        #    length_scale=length_scale, length_scale_bounds="fixed", nu=4.5
        # )
        kernel = ConstantKernel(10.0, constant_value_bounds=(0.1, 1e13)) * RBF(
            length_scale=length_scale, length_scale_bounds="fixed"
        )
    else:
        # kernel = ConstantKernel(10.0, constant_value_bounds=(0.1, 1e13)) * Matern(
        #    length_scale=200.0, length_scale_bounds=(1, 1e4), nu=4.5
        # )
        kernel = ConstantKernel(10.0, constant_value_bounds=(0.1, 1e13)) * RBF(
            length_scale=200.0, length_scale_bounds=(1, 1e4)
        )

    gaussian_process = GaussianProcessRegressor(
        kernel=kernel, n_restarts_optimizer=20, alpha=varia**2
    )
    ret = gaussian_process.fit(points, vals)
    return ret


def chisqr(exp, uncert, pred):
    return np.sum((exp - pred) ** 2 / uncert)


def getPrediction(histogram, gp):
    bc = histogram.axes[0].centers
    v = gp.predict(bc.reshape(-1, 1))
    cs = chisqr(histogram.values(), histogram.variances(), v)
    print(f"Reduced Chi^2 results are {cs:0.2f}; Chi^2/DOF = {cs/(len(bc) - 1)}")
    h = np.histogram(bc, weights=v, bins=histogram.axes[0].edges)
    ratio = histogram.sum().value / sum(h[0])
    h = (h[0] * ratio, h[1])
    print(f"Scaling is {ratio}")
    return h


def plotGaussianProcess(ax, gp, min_bound=1050):
    ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    mean_prediction, std_prediction = gp.predict(ls, return_std=True)
    ax.plot(ls.ravel(), mean_prediction, label="Mean prediction", color="tab:orange")
    ax.fill_between(
        ls.ravel(),
        mean_prediction - std_prediction,
        mean_prediction + std_prediction,
        alpha=0.3,
        # label=r"95% confidence interval",
        color="tab:orange",
    )
    return ax


def generatePulls(
    histogram,
    gp,
    outdir=Path("diagnostics"),
    min_bound=1050,
    target=None,
    sig_hist=None,
    sig_name=None,
    sig_strength=None,
    name="pulls_bkg_only",
):
    bc = histogram.axes[0].centers
    v, std = gp.predict(bc.reshape(-1, 1), return_std=True)
    h = np.histogram(bc, weights=v, bins=histogram.axes[0].edges)
    pred = PlotObject((*h, std))
    obs = PlotObject(histogram)
    # fig, ax = drawAs1DHist(obs, yerr=True, fill=False)
    fig, ax = drawAsScatter(obs, yerr=True)
    # drawAs1DHist(ax, pred, yerr=True, fill=False)
    plotGaussianProcess(ax, gp, min_bound=min_bound)
    ax.tick_params(axis="x", labelbottom=False)
    addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    if sig_hist:
        drawAs1DHist(ax, PlotObject(sig_hist), yerr=True, fill=None)
    ax.set_yscale("linear")
    ab = ax.bottom_axes[0]
    drawPull(ab, pred, obs, hline_list=[-1, 0, 1])
    # ab.set_ylabel(r"$\frac{obs - pred}{\sqrt{\sigma_{p}^2 + \sigma_{o}^2}}$")
    ab.set_ylabel(r"$\frac{obs - pred}{\sigma_{o}}$")
    if target:
        ax.axvline(target, color="black", linewidth=0.3, linestyle="-.")
        ab.axvline(target, color="black", linewidth=0.3, linestyle="-.")
    addEra(ax, 137)
    addPrelim(ax, additional_text=f"\n$\\lambda_{{312}}''$ ")
    addTitles1D(ax, histogram, top_pad=0.2)
    ax.legend()
    transform = ax.transAxes
    cs = chisqr(histogram.values(), histogram.variances(), v)
    scale = gp.kernel_.get_params()["k2__length_scale"]
    text = f"$\chi^2/DOF = {cs/(len(bc)-1):0.2f}$\nLength Scale={scale:0.2f}"
    if sig_name:
        text += f"\n{sig_name}: r={sig_strength}"

    ax.text(
        0.95,
        0.7,
        text,
        horizontalalignment="right",
        transform=transform,
        fontsize=20,
    )
    outdir.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf")


def generateDiagnosticPlots(
    histogram,
    gp,
    min_bound=1050,
    outdir=Path("diagnostics"),
    name="gaussian_process_fit",
):
    ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    fig, ax = plt.subplots()
    mean_prediction, std_prediction = gp.predict(ls, return_std=True)
    bc = histogram.axes[0].centers
    v = gp.predict(bc.reshape(-1, 1))
    vals, points, varia = (
        histogram.values(),
        histogram.axes[0].centers,
        np.sqrt(histogram.variances()),
    )
    ax.errorbar(
        points,
        vals,
        varia,
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observations",
    )
    ax.plot(ls.ravel(), mean_prediction, label="Mean prediction", color="tab:orange")
    ax.fill_between(
        ls.ravel(),
        mean_prediction - std_prediction,
        mean_prediction + std_prediction,
        alpha=0.3,
        color="tab:orange",
    )
    transform = ax.transAxes
    cs = chisqr(histogram.values(), histogram.variances(), v)
    scale = gp.kernel_.get_params()["k2__length_scale"]
    ax.text(
        0.95,
        0.7,
        f"$\chi^2/DOF = {cs/(len(bc)-1):0.2f}$\nLength Scale={scale:0.2f}",
        horizontalalignment="right",
        transform=transform,
    )
    ax.legend()
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel("Events")
    outdir.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(outdir / f"{name}.pdf")

    fig, ax = plt.subplots()
    bc = histogram.axes[0].centers
    v = gp.predict(bc.reshape(-1, 1))
    v, e = np.histogram(bc, weights=v, bins=histogram.axes[0].edges)
    ax.bar(e[:-1], v, width=np.diff(e), edgecolor="black", align="edge")
    ax.plot(ls, mean_prediction, label="Mean prediction")
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(outdir / f"{name}_template_shape.pdf")


def getMatching(ax, val):
    return next(x for x in ax if val in x)


def parseArgs():
    parser = argparse.ArgumentParser(
        description="Generate template shape output files for fitting"
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("--inject-name", type=str, default=None)
    parser.add_argument("--inject-strength", type=float, default=None)
    parser.add_argument("--inject-fake", nargs=2, type=float, default=None)
    parser.add_argument("--force-scale", type=float, default=None)
    parser.add_argument("--window", type=float, nargs=2, default=None)
    parser.add_argument("-r", "--input-control-region", default=None, type=str)
    parser.add_argument("-o", "--output", required=True, type=str)
    parser.add_argument(
        "-p", "--plot-dir", default=None, type=str, help="If set, save diagnostic plots"
    )
    parser.add_argument(
        "-n",
        "--name",
        required=True,
        type=str,
        help="Name of the histogram to use for the fit",
    )
    parser.add_argument(
        "-b",
        "--backgrounds",
        required=True,
        type=str,
        nargs="+",
        help="Backgrounds to add",
    )
    parser.add_argument("--only-cr", action="store_true")
    parser.add_argument(
        "-s",
        "--signals",
        required=True,
        type=str,
        nargs="+",
        help="Signals to add to output",
    )
    parser.add_argument(
        "--rebin",
        type=int,
        default=1,
        help="Rebinning",
    )
    parser.add_argument("-m", "--min-bound", default=1050, type=float)
    args = parser.parse_args()
    return args


def addSignalToHist(hist, normalize_to, mean, sigma, name="fake_signal"):
    norm = hist[{"dataset": normalize_to}].sum().value
    gaus = np.random.normal(loc=mean, scale=sigma, size=10000)
    hist.fill(name, gaus, weight=norm / 10000)
    return hist


def main():
    args = parseArgs()
    data = pkl.load(open(args.input, "rb"))
    plotdir = Path(args.plot_dir)
    root_output = uproot.recreate(args.output)
    histos = data["histograms"]
    histogram = histos[args.name]
    target = None
    sigh = None
    signame=args.inject_name
    append_to_files = (
        f"_{signame}_r{args.inject_strength}"
        if signame
        else "_bkg_only"
    )
    if args.force_scale:
        append_to_files += f"__scale_force_{round(args.force_scale)}"
    window=args.window
    if window:
        append_to_files += f"_w{window[0]}_{window[1]}"
    append_to_files = append_to_files.replace(".", "p")
    if args.only_cr:
        for bkg in args.backgrounds:
            if args.input_control_region:
                print("Optimizing length scale in control region")
                datacr = pkl.load(open(args.input_control_region, "rb"))
                hcr = datacr["histograms"][args.name]
                hcr = hcr[
                    getMatching(hcr.axes[0], bkg),
                    hist.loc(args.min_bound) :: hist.rebin(args.rebin),
                ]
                crgp = fitGp(hcr)
                ls = crgp.kernel_.get_params()["k2__length_scale"]
                #generatePulls(
                #    hcr,
                #    crgp,
                #    plotdir / bkg,
                #    args.min_bound,
                #    target=target,
                #    sig_hist=sigh,
                #    sig_name=signame,
                #    sig_strength=args.inject_strength,
                #    name="pulls_cr" + append_to_files,
                #)
                # generateDiagnosticPlots(
                #    hcr, crgp, args.min_bound, plotdir / bkg, name="cr_fit" + append_to_files
                # )
        return 0
    for bkg in args.backgrounds:
        h = histogram[
            getMatching(histogram.axes[0], bkg),
            hist.loc(args.min_bound) :: hist.rebin(args.rebin),
        ]
        if signame and args.inject_strength:
            hname = signame
            if args.inject_fake:
                m, s = args.inject_fake
                print(f"Using fake signal")
                hname += f"_fake_s{s}"
                append_to_files += hname
                append_to_files = append_to_files.replace(".", "p")
                histogram = addSignalToHist(histogram, signame, m, s, hname)
            target = int(signame.split("_")[2])
            signame = hname
            print(f"Injecting signal {signame} at rate {args.inject_strength}")
            sigh = (
                args.inject_strength
                * histogram[hname, hist.loc(args.min_bound) :: hist.rebin(args.rebin)]
            )
            h += sigh

        if args.input_control_region:
            print("Optimizing length scale in control region")
            datacr = pkl.load(open(args.input_control_region, "rb"))
            hcr = datacr["histograms"][args.name]
            hcr = hcr[
                getMatching(hcr.axes[0], bkg),
                hist.loc(args.min_bound) :: hist.rebin(args.rebin),
            ]
            crgp = fitGp(hcr)
            ls = crgp.kernel_.get_params()["k2__length_scale"]
            #generatePulls(
            #    hcr,
            #    crgp,
            #    plotdir / bkg,
            #    args.min_bound,
            #    target=target,
            #    sig_hist=sigh,
            #    sig_name=signame,
            #    sig_strength=args.inject_strength,
            #    name="pulls_cr" + append_to_files,
            #)
            generateDiagnosticPlots(
               hcr, crgp, args.min_bound, plotdir / bkg, name="cr_fit" + append_to_files
            )
            print(f"Found length scale {ls:0.2f}")
            gp = fitGp(h, ls, window=window)
        elif args.force_scale:
            print(f"Using manually entered length scale {args.force_scale}")
            gp = fitGp(h, args.force_scale, window=window)
        else:
            gp = fitGp(h, window=window)
            print("Optimizing length scale in signal region")
            ls = gp.kernel_.get_params()["k2__length_scale"]
            print(f"Found length scale {ls:0.2f}")

        res = getPrediction(h, gp)
        # generateDiagnosticPlots(h, gp, args.min_bound, plotdir / bkg, name="gaussian_process_fit" + append_to_files)
        # sigh=None
        generatePulls(
            h,
            gp,
            plotdir / bkg,
            args.min_bound,
            target=target,
            sig_hist=sigh,
            sig_name=signame,
            sig_strength=args.inject_strength,
            name="pull_sr" + append_to_files,
        )
        root_output[f"SignalChannel/{bkg}/nominal"] = res

    for bkg in args.backgrounds:
        h = histogram[
            getMatching(histogram.axes[0], bkg),
            hist.loc(args.min_bound) :: hist.rebin(args.rebin),
        ]
        root_output["data_obs"] = h

    for sig in args.signals:
        for x in (y for y in histogram.axes[0] if sig in y):
            h = histogram[x, hist.loc(args.min_bound) :]
            root_output[f"SignalChannel/{x}/nominal"] = h


if __name__ == "__main__":
    main()
