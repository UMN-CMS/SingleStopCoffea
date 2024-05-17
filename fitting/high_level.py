import argparse
import contextlib
import itertools as it
import math
import pickle as pkl
import shutil
import random
import sys
from collections import namedtuple
from pathlib import Path

import numpy as np
import yaml

import analyzer.file_utils as futil
import analyzer.plotting as plotting
import gpytorch
import hist
import linear_operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import uhi
from analyzer.plotting.mplstyles import loadStyles
from analyzer.plotting.utils import subplots_context
from gpytorch.kernels import ScaleKernel as SK
from matplotlib.patches import Polygon
from rich.progress import Progress, track

from . import models, regression
from .plot_tools import createSlices, getPolyFromSquares, makeSquares, simpleGrid

torch.set_default_dtype(torch.float64)


def saveDiagnosticPlots(plots, dirdata, save_dir):
    for name, (fig, ax) in plots.items():
        o = (save_dir / name).with_suffix(".pdf")
        fig.savefig(o)


def makeDiagnosticPlots(pred, raw_test, raw_train, raw_hist, mask=None):
    ret = {}
    if mask is not None:
        squares = makeSquares(raw_test.X[mask], raw_test.E)
        points = getPolyFromSquares(squares)

    def addWindow(ax):
        if mask is None:
            return
        else:
            poly = Polygon(points, edgecolor="green", linewidth=3, fill=False)
            ax.add_patch(poly)

    pred_mean = pred.Y
    pred_variances = pred.V

    all_x2 = (pred_mean - raw_test.Y) ** 2 / raw_test.V
    x2 = torch.sum(all_x2)

    fig, ax = plt.subplots(layout="tight")
    simpleGrid(ax, raw_test.E, raw_train.X, raw_train.Y)
    ax.set_title("Masked Inputs (Training)")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["training_points"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, pred_mean)
    ax.set_title("GPR Mean Prediction")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["gpr_mean"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    simpleGrid(ax, raw_test.E, raw_test.X, raw_test.Y)
    ax.set_title("Observed Outputs")
    addWindow(ax)
    ret["observed_outputs"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, raw_test.V)
    ax.set_title("Observed Variances")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["observed_variances"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(ax, raw_test.E, raw_test.X, pred.V)
    ax.set_title("Pred Variances")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    addWindow(ax)
    ret["predicted_variances"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(
        ax,
        raw_test.E,
        raw_test.X,
        (raw_test.Y - pred_mean) / torch.sqrt(pred.V),
        cmap="coolwarm",
    )
    f.set_clim(-5, 5)
    ax.set_title("Pulls")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    ax.cax.set_ylabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
    addWindow(ax)
    ret["pulls_pred"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(
        ax,
        raw_test.E,
        raw_test.X,
        (raw_test.Y - pred_mean) / torch.sqrt(raw_test.V),
        cmap="coolwarm",
    )
    f.set_clim(-5, 5)
    ax.set_title("Pulls")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    ax.cax.set_ylabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{o}}$")
    addWindow(ax)
    ret["pulls_obs"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    all_pulls = (raw_test.Y - pred_mean) / torch.sqrt(pred.V)
    p = plotting.PlotObject.fromHist(
        uhi.numpy_plottable.ensure_plottable_histogram(
            np.histogram(all_pulls[torch.abs(all_pulls) < np.inf], bins=20)
        )
    )
    plotting.drawAs1DHist(ax, p, yerr=False)
    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
    ax.set_ylabel("Count")
    ret["pulls_hist"] = (fig, ax)

    return ret


def makeSlicePlots(pred, test_data, hist, window, dim, save_dir, dirdata):
    pred_mean, _ = regression.pointsToGrid(test_data.X, pred.Y, test_data.E)
    pred_var, _ = regression.pointsToGrid(test_data.X, pred.V, test_data.E)
    obs_vals, _ = regression.pointsToGrid(test_data.X, test_data.Y, test_data.E)
    obs_vars, filled = regression.pointsToGrid(test_data.X, test_data.V, test_data.E)

    (save_dir / "slices" / f"along_{dim}").mkdir(parents=True, exist_ok=True)

    for val, f, ax in createSlices(
        pred_mean.hist,
        pred_var.hist,
        obs_vals.hist,
        obs_vars.hist,
        test_data.E,
        filled,
        observed_title="CRData",
        mask_function=window,
        just_window=window is not None,
        slice_dim=dim,
    ):
        plotting.addTitles1D(
            ax, plotting.PlotObject.fromHist(hist[{hist.axes[dim].name: sum}])
        )
        o = (
            save_dir
            / "slices"
            / f"along_{dim}"
            / (f"slice_{round(float(val),3)}".replace(".", "p") + ".pdf")
        )
        f.savefig(o)
        dirdata.set(o, dict(slice_dim=dim, val=float(val)))
        plt.close(f)


def doCompleteRegression(
    inhist,
    window_func,
    dir_data,
    kernel=None,
    model_maker=None,
    rebin=1,
):

    save_dir = dir_data.directory

    train_data = regression.makeRegressionData(
        inhist[hist.rebin(rebin), hist.rebin(rebin)], window_func, exclude_less=0.001
    )
    test_data = regression.makeRegressionData(
        inhist[hist.rebin(rebin), hist.rebin(rebin)], None, exclude_less=0.001
    )
    train_transform = regression.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)

    with subplots_context() as (fig, ax):
        out = save_dir / "original.pdf"
        plotting.drawAs2DHist(ax, plotting.PlotObject.fromHist(inhist))
        fig.savefig(out)

    with subplots_context(1, 2, figsize=(10, 5)) as (fig, ax):
        out = save_dir / "masked_area.pdf"
        simpleGrid(ax[0], test_data.E, test_data.X, test_data.Y)
        simpleGrid(ax[1], train_data.E, train_data.X, train_data.Y)
        fig.savefig(out)

    use_cuda = True

    if torch.cuda.is_available() and use_cuda:
        print("USING GPU")
        train = regression.sendToGpu(normalized_train_data)
    else:
        train = normalized_train_data

    lr = 0.05
    ok = False
    while not ok:
        model, likelihood = regression.createModel(
            train, kernel=kernel, model_maker=model_maker, learn_noise=False
        )
        if torch.cuda.is_available() and use_cuda:
            model = model.cuda()
            likelihood = likelihood.cuda()

        try:
            model, likelihood, evidence = regression.optimizeHyperparams(
            model,
            likelihood,
            train,
            bar=False,
            iterations=800,
            lr=0.05,
            get_evidence=True)
            ok=True
        except linear_operator.utils.errors.NanError as e:
            lr = lr + random.random()/100
            print(f"CHOLESKY FAILED: retrying with lr={round(lr,3)}")
            print(e)

    print("Done training")
    if torch.cuda.is_available() and use_cuda:
        model = model.cpu()
        likelihood = likelihood.cpu()
    params = list(model.named_parameters_and_constraints())
    pred = regression.getPrediction(model, likelihood, normalized_test_data)
    pred = train_transform.iTransform(
        regression.DataValues(
            normalized_test_data.X, pred.mean, pred.variance, normalized_test_data.E
        )
    )

    data = {
        "evidence": evidence,
        "model_string": str(model),
    }

    if window_func:
        mask = regression.getBlindedMask(
            pred.X, pred.Y, test_data.Y, test_data.V, window_func
        )
        bpred_mean = pred.Y[mask]
        obs_mean = test_data.Y[mask]
        obs_var = test_data.V[mask]
        chi2 = torch.sum((obs_mean - bpred_mean) ** 2 / obs_var) / torch.count_nonzero(
            mask
        )
        avg_pull = torch.sum(
            torch.abs((obs_mean - bpred_mean)) / torch.sqrt(obs_var)
        ) / torch.count_nonzero(mask)

        data.update(
            {
                "chi2_blinded": float(chi2),
                "avg_abs_pull": float(avg_pull),
            }
        )
        print(f"Chi^2/bins = {chi2}")
        print(f"Avg Abs pull = {avg_pull}")
    else:
        mask = None
    diagnostic_plots = makeDiagnosticPlots(pred, test_data, train_data, inhist, mask)
    saveDiagnosticPlots(diagnostic_plots, dir_data, save_dir)
    makeSlicePlots(pred, test_data, inhist, window_func, 0, save_dir, dir_data)
    makeSlicePlots(pred, test_data, inhist, window_func, 1, save_dir, dir_data)

    save_data = dict(
        train_data=train_data,
        test_data=test_data,
        prediction=pred,
        model_state=model.state_dict(),
        model=model,
    )

    torch.save(save_data, save_dir / "train_model.pth")
    # torch.save(model.state_dict(), save_dir / "train_model.pth")

    dir_data.setGlobal(data)


def scan(hist, kernel, window_func_generator, base_dir, kernel_name=""):
    base_dir = Path(base_dir)
    inducing_ratio = 2

    def mm(train_x, train_y, likelihood, kernel, **kwargs):
        return models.InducingPointModel(
            train_x, train_y, likelihood, kernel, inducing=train_x[::inducing_ratio]
        )

    i=0
    for name, wf, data in window_func_generator:
        i = i + 1
        print("=" * 20)
        if kernel_name:
            print(f"KERNEL: {kernel_name}")
        print(f"Now processing area {name}")

        path = base_dir / name
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)
        dirdata = futil.DirectoryData(path)

        dirdata.setGlobal({"window": data, "inducing_ratio": inducing_ratio})
        doCompleteRegression(hist, wf, dirdata, model_maker=mm, kernel=kernel)
        plt.close("all")
        print("=" * 20)


def main():
    from analyzer.core import AnalysisResult
    from analyzer.datasets import SampleManager

    mpl.use("Agg")

    res = AnalysisResult.fromFile("results/data_control.pkl")
    sample_manager = SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")
    bkg_name = "CR0b_Data2018"
    hists = res.getMergedHistograms(sample_manager)
    complete_hist = hists["ratio_m14_vs_m24"]
    orig = complete_hist[
        ..., hist.loc(1000) : hist.loc(3000), hist.loc(0.35) : hist.loc(1.0)
    ]
    narrowed = orig[..., :: hist.rebin(1), :: hist.rebin(1)]
    qcd_hist = narrowed[bkg_name, ...]
    qcd_hist = narrowed[bkg_name, ...]

    x_iter = map(float, [1200, 1500, 2000])
    x_size_iter = map(float, [100, 150])
    y_iter = map(float, [0.5, 0.7])
    y_size_iter = map(float, [0.05, 0.07])
    generator = list(
        (
            f"E_{round(x)}_{round(y,2)}_{round(a)}_{round(b,3)}".replace(".", "p"),
            regression.ellipseMasker(torch.tensor([x, y]), a, b),
            {"x": x, "y": y, "a": a, "b": b},
        )
        for x, a, y, b in it.product(x_iter, x_size_iter, y_iter, y_size_iter)
    )
    generator = generator + [("Complete", None, {})]

    nnrbf256 = SK(models.NNRBFKernel(odim=2, layer_sizes=(256, 128, 16)))
    nnrbf1024 = SK(models.NNRBFKernel(odim=2, layer_sizes=(1024, 1024, 16)))
    nnrbf32 = SK(models.NNRBFKernel(odim=2, layer_sizes=(32, 16)))
    nnrbf32_16_8 = SK(models.NNRBFKernel(odim=2, layer_sizes=(32, 16, 8)))

    nnrbf_1000_500_50 = SK(models.NNRBFKernel(odim=1, layer_sizes=(1000, 500, 50)))

    nnrbf16 = SK(models.NNRBFKernel(odim=2, layer_sizes=(16, 8)))

    nnrq256 = SK(models.NNRQKernel(odim=2, layer_sizes=(256, 128, 16)))
    nnrq32 = SK(models.NNRQKernel(odim=2, layer_sizes=(32, 16)))

    nnsmk_8_8 = models.NNSMKernel(odim=2, layer_sizes=(8, 8), num_mixtures=4)
    nnsmk_32_16_8 = models.NNSMKernel(odim=2, layer_sizes=(32, 16, 8), num_mixtures=4)
    smk_4 = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=4, ard_num_dims=2)

    grbf = SK(models.GeneralRBF(ard_num_dims=2))
    grq = SK(models.GeneralRQ(ard_num_dims=2))

    rbf = SK(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    cosine = SK(gpytorch.kernels.CosineKernel(ard_num_dims=2))
    kernels = {
        "grbf": grbf,
        "rbf": rbf,
        "nnrbf_32_16_8": nnrbf32_16_8,
        # "nnrbf_1024_1024_16": nnrbf1024,
        #"nnrbf_256_128_16": nnrbf256,
        #"nnrbf_1000_500_50": nnrbf_1000_500_50,
        # "nnrbf_16_8": nnrbf16,
        # "nnrq_32_16": nnrq32,
        # "nnrq_256_128_16": nnrq256,
        # "grq": grq,
        # "cosine" : cosine
        #"nnsmk_32_16_8": nnsmk_32_16_8,
        #"nnsmk_8_8": nnsmk_8_8,
        # "smk_4": smk_4,
    }

    p = Path("allscans")

    for n, k in kernels.items():
        scan(qcd_hist, k, generator, p / n, kernel_name=n)


if __name__ == "__main__":
    loadStyles()
    main()
