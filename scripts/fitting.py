import pickle as pkl
import uproot
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


def chisqr(exp,uncert, pred):
    return np.sum((exp-pred)**2 / uncert)

def getPrediction(histogram, gp):
    bc = histogram.axes[0].centers
    v = gp.predict(bc.reshape(-1, 1))
    cs = chisqr(histogram.values(), histogram.variances(), v)
    print(f"Reduced Chi^2 results are {cs:0.2f}; Chi^2/DOF = {cs/(len(bc) - 1)}")
    return np.histogram(bc, weights=v, bins=histogram.axes[0].edges)


def generateDiagnosticPlots(histogram, gp, min_bound=1050):
    ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    fig, ax = plt.subplots()
    mean_prediction, std_prediction = gaussian_process.predict(ls, return_std=True)
    ax.plot(points, vals, label=histogram.name, linestyle="dotted")
    ax.errorbar(
        points,
        vals,
        v,
        linestyle="None",
        color="tab:blue",
        marker=".",
        markersize=10,
        label="Observations",
    )
    ax.plot(ls, mean_prediction, label="Mean prediction")
    ax.fill_between(
        ls.ravel(),
        mean_prediction - 1.96 * std_prediction,
        mean_prediction + 1.96 * std_prediction,
        alpha=0.5,
        label=r"95% confidence interval",
    )
    ax.legend()
    ax.xlabel(histogram.name)
    ax.ylabel("Events")


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
        root_output[bkg] = res

    for bkg in args.backgrounds:
        h = histogram[getMatching(histogram.axes[0], bkg), hist.loc(args.min_bound) :]
        root_output["data_obs"] = h

    for sig in args.signals:
        h = histogram[sig, hist.loc(args.min_bound) :]
        root_output[sig] = h


if __name__ == "__main__":
    main()
