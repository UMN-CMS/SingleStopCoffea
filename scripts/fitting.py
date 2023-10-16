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


# class RBF(Kernel):
#     def __init__(
#         self,
#         length_scale=1.0,
#         length_scale_bounds=(1e-5, 1e5),
#         peak_location=1,
#         peak_location_bound=(1e-5, 1e5),
#     ):
#         self.length_scale = length_scale
#         self.length_scale_bounds = length_scale_bounds

#         self.peak_location = peak_location
#         self.peak_location_bounds = peak_location_bounds

#     @property
#     def anisotropic(self):
#         return np.iterable(self.length_scale) and len(self.length_scale) > 1

#     @property
#     def hyperparameter_length_scale(self):
#         if self.anisotropic:
#             return Hyperparameter(
#                 "length_scale",
#                 "numeric",
#                 self.length_scale_bounds,
#                 len(self.length_scale),
#             )
#         return Hyperparameter("length_scale", "numeric", self.length_scale_bounds)

#     @property
#     def hyperparameter_peak_location(self):
#         return Hyperparameter(
#             "peak_location",
#             "numeric",
#             self.peak_location_bounds,
#             len(self.peak_location),
#         )

#     def __call__(self, X, Y=None, eval_gradient=False):
#         """Return the kernel k(X, Y) and optionally its gradient.

#         Parameters
#         ----------
#         X : ndarray of shape (n_samples_X, n_features)
#             Left argument of the returned kernel k(X, Y)

#         Y : ndarray of shape (n_samples_Y, n_features), default=None
#             Right argument of the returned kernel k(X, Y). If None, k(X, X)
#             if evaluated instead.

#         eval_gradient : bool, default=False
#             Determines whether the gradient with respect to the log of
#             the kernel hyperparameter is computed.
#             Only supported when Y is None.

#         Returns
#         -------
#         K : ndarray of shape (n_samples_X, n_samples_Y)
#             Kernel k(X, Y)

#         K_gradient : ndarray of shape (n_samples_X, n_samples_X, n_dims), \
#                 optional
#             The gradient of the kernel k(X, X) with respect to the log of the
#             hyperparameter of the kernel. Only returned when `eval_gradient`
#             is True.
#         """
#         X = np.atleast_2d(X)
#         length_scale = _check_length_scale(X, self.length_scale)

#         def metric(x, y):
#             return cdist(
#                 x / self.length_scale, self.peak_location / self.length_scale
#             ) + cdist(y / self.length_scale, self.peak_location / self.length_scale)

#         if Y is None:
#             dists = pdist(X, metric=metric)
#             K = np.exp(-0.5 * dists)
#             K = squareform(K)
#             np.fill_diagonal(K, 1)
#         else:
#             if eval_gradient:
#                 raise ValueError("Gradient can only be evaluated when Y is None.")
#             dists = cdist(X, Y, metric=metric)
#             K = np.exp(-0.5 * dists)

#         if eval_gradient:
#             if (
#                 self.hyperparameter_length_scale.fixed
#                 and self.hyperparameter_peak_location.fixed
#             ):
#                 # Hyperparameter l kept fixed
#                 return K, np.empty((X.shape[0], X.shape[0], 0))
#             elif not self.anisotropic or length_scale.shape[0] == 1:
#                 K_gradient = (K * squareform(dists))[:, :, np.newaxis]
#                 return K, K_gradient
#             elif self.anisotropic:
#                 # We need to recompute the pairwise dimension-wise distances
#                 K_gradient = (X[:, np.newaxis, :] - X[np.newaxis, :, :]) ** 2 / (
#                     length_scale**2
#                 )
#                 K_gradient *= K[..., np.newaxis]
#                 return K, K_gradient
#         else:
#             return K

#     def __repr__(self):
#         if self.anisotropic:
#             return "{0}(length_scale=[{1}])".format(
#                 self.__class__.__name__,
#                 ", ".join(map("{0:.3g}".format, self.length_scale)),
#             )
#         else:  # isotropic
#             return "{0}(length_scale={1:.3g})".format(
#                 self.__class__.__name__, np.ravel(self.length_scale)[0]
#             )


def fitGp(histogram, length_scale=None):
    vals, points, varia = (
        histogram.values(),
        histogram.axes[0].centers.reshape(-1, 1),
        np.sqrt(histogram.variances()),
    )
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
    ax.set_yscale("log")
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
        fontsize=12,
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


def main():
    parser = argparse.ArgumentParser(
        description="Generate template shape output files for fitting"
    )
    parser.add_argument("-i", "--input", required=True, type=str)
    parser.add_argument("--inject-name", type=str, default=None)
    parser.add_argument("--inject-strength", type=float, default=None)
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
    data = pkl.load(open(args.input, "rb"))
    plotdir = Path(args.plot_dir)
    root_output = uproot.recreate(args.output)
    histos = data["histograms"]
    histogram = histos[args.name]
    target = None
    sigh = None
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
                generatePulls(
                    hcr,
                    crgp,
                    plotdir / bkg,
                    args.min_bound,
                    target=target,
                    sig_hist=sigh,
                    sig_name=args.inject_name,
                    sig_strength=args.inject_strength,
                    name="pulls_cr",
                )
                generateDiagnosticPlots(
                    hcr, crgp, args.min_bound, plotdir / bkg, name="cr_fit"
                )
        return 0
    for bkg in args.backgrounds:
        h = histogram[
            getMatching(histogram.axes[0], bkg),
            hist.loc(args.min_bound) :: hist.rebin(args.rebin),
        ]
        if args.inject_name and args.inject_strength:
            target = int(args.inject_name.split("_")[2])
            print(f"Injecting signal {args.inject_name} at rate {args.inject_strength}")

            sigh = (
                args.inject_strength
                * histogram[
                    args.inject_name, hist.loc(args.min_bound) :: hist.rebin(args.rebin)
                ]
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
            generatePulls(
                hcr,
                crgp,
                plotdir / bkg,
                args.min_bound,
                target=target,
                sig_hist=sigh,
                sig_name=args.inject_name,
                sig_strength=args.inject_strength,
                name="pulls_cr",
            )
            generateDiagnosticPlots(
                hcr, crgp, args.min_bound, plotdir / bkg, name="cr_fit"
            )
            print(f"Found length scale {ls:0.2f}")
            gp = fitGp(h, ls)
        else:
            gp = fitGp(h)
            print("Optimizing length scale in signal region")
            ls = gp.kernel_.get_params()["k2__length_scale"]
            print(f"Found length scale {ls:0.2f}")

        res = getPrediction(h, gp)
        generateDiagnosticPlots(h, gp, args.min_bound, plotdir / bkg)
        # sigh=None
        generatePulls(
            h,
            gp,
            plotdir / bkg,
            args.min_bound,
            target=target,
            sig_hist=sigh,
            sig_name=args.inject_name,
            sig_strength=args.inject_strength,
            name="pull_sr"
            + (
                f"_{args.inject_name}_r{args.inject_strength}".replace(".", "p")
                if args.inject_name
                else "_bkg_only"
            ),
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
