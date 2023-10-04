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
)
import re
from analyzer.plotting.core_plots import loadStyles
loadStyles()



def fitGp(histogram, length_scale=None):
    vals, points, varia = (
        histogram.values(),
        histogram.axes[0].centers.reshape(-1, 1),
        np.sqrt(histogram.variances()),
    )
    if length_scale:
        kernel = ConstantKernel(10.0, constant_value_bounds=(0.1, 1e13)) * RBF(
            length_scale=length_scale, length_scale_bounds="fixed"
        )
    else:
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
    h =  np.histogram(bc, weights=v, bins=histogram.axes[0].edges)
    ratio =  histogram.sum().value/ sum(h[0]) 
    h =  (h[0] * ratio,h[1])
    print(f"Scaling is {ratio}")
    return h


def generateDiagnosticPlots(histogram, gp, min_bound=1050, outdir=Path("diagnostics")):
    ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    fig, ax = plt.subplots()
    mean_prediction, std_prediction = gp.predict(ls, return_std=True)

    bc = histogram.axes[0].centers
    v = gp.predict(bc.reshape(-1, 1))
    cs = chisqr(histogram.values(), histogram.variances(), v)
    vals, points, varia = (
        histogram.values(),
        histogram.axes[0].centers,
        np.sqrt(histogram.variances()),
    )

    # ax.plot(points, vals, label=histogram.name, linestyle="dotted")
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
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.3,
        label=r"95% confidence interval",
        color="tab:orange",
    )
    transform = ax.transAxes

    scale = gp.kernel_.get_params()["k2__length_scale"]
    ax.text(0.95, 0.7, f"$\chi^2/DOF = {cs/(len(bc)-1):0.2f}$\nLength Scale={scale:0.2f}", horizontalalignment="right", transform=transform)
    ax.legend()
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel("Events")
    outdir.mkdir(exist_ok=True, parents=True)
    fig.tight_layout()
    fig.savefig(outdir / "gaussian_process_fit.pdf")

    fig, ax = plt.subplots()
    bc = histogram.axes[0].centers
    v = gp.predict(bc.reshape(-1, 1))
    v, e = np.histogram(bc, weights=v, bins=histogram.axes[0].edges)
    ax.bar(e[:-1], v, width=np.diff(e), edgecolor="black", align="edge")
    ax.plot(ls, mean_prediction, label="Mean prediction")
    ax.set_xlabel(histogram.axes[0].label)
    ax.set_ylabel("Events")
    fig.tight_layout()
    fig.savefig(outdir / "gaussian_template_shape.pdf")


def getMatching(ax, val):
    return next(x for x in ax if val in x)


def main():
    parser = argparse.ArgumentParser(
        description="Generate template shape output files for fitting"
    )
    parser.add_argument("-i", "--input", required=True, type=str)
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
    parser.add_argument(
        "-s",
        "--signals",
        required=True,
        type=str,
        nargs="+",
        help="Signals to add to output",
    )
    parser.add_argument("-m", "--min-bound", default=1050, type=float)
    args = parser.parse_args()
    data = pkl.load(open(args.input, "rb"))
    plotdir = Path(args.plot_dir)
    root_output = uproot.recreate(args.output)
    histos = data["histograms"]
    histogram = histos[args.name]
    for bkg in args.backgrounds:
        h = histogram[getMatching(histogram.axes[0], bkg), hist.loc(args.min_bound) :]
        if args.input_control_region:
            print("Optimizing length scale in control region")
            datacr = pkl.load(open(args.input_control_region, "rb"))
            hcr = datacr["histograms"][args.name]
            hcr = hcr[getMatching(hcr.axes[0], bkg), hist.loc(args.min_bound) :]
            crgp = fitGp(hcr)
            ls = crgp.kernel_.get_params()["k2__length_scale"]
            print(f"Found length scale {ls:0.2f}")
            gp = fitGp(h, ls)
        else:
            gp = fitGp(h)
            print("Optimizing length scale in signal region")
            ls = gp.kernel_.get_params()["k2__length_scale"]
            print(f"Found length scale {ls:0.2f}")

        res = getPrediction(h, gp)
        generateDiagnosticPlots(h, gp, args.min_bound, plotdir / bkg)
        root_output[f"SignalChannel/{bkg}/nominal"] = res

    for bkg in args.backgrounds:
        h = histogram[getMatching(histogram.axes[0], bkg), hist.loc(args.min_bound) :]
        root_output["data_obs"] = h

    for sig in args.signals:
        for x in (y for y in histogram.axes[0] if sig in y):
            h = histogram[x, hist.loc(args.min_bound) :]
            root_output[f"SignalChannel/{x}/nominal"] = h


if __name__ == "__main__":
    main()
