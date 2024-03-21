import contextlib
import itertools as it
import math
import pickle as pkl
import shutil
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
import matplotlib.pyplot as plt
import torch
import uhi
from analyzer.plotting.mplstyles import loadStyles
from analyzer.plotting.utils import subplots_context
from gpytorch.kernels import ScaleKernel as SK
from rich.progress import Progress, track
from matplotlib.patches import Polygon


from . import models, regression
from .plot_tools import createSlices, simpleGrid, makeSquares, getPolyFromSquares

torch.set_default_dtype(torch.float64)


def saveDiagnosticPlots(plots, dirdata, save_dir):
    for name, (fig, ax) in plots.items():
        data = data or {}
        o = (save_dir / name).with_extension(".pdf")
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
            poly = Polygon(points, edgecolor="red", fill=False)
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
        ax, raw_test.E, raw_test.X, (raw_test.Y - pred_mean) / torch.sqrt(pred.V)
    )
    f.set_clim(-5, 5)
    ax.set_title("Pulls")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    ax.cax.set_ylabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
    addWindow(ax)
    ret["pulls_pred"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    f = simpleGrid(
        ax, raw_test.E, raw_test.X, (raw_test.Y - pred_mean) / torch.sqrt(raw_test.V)
    )
    f.set_clim(-5, 5)
    ax.set_title("Pulls")
    plotting.addTitles2D(ax, plotting.PlotObject.fromHist(raw_hist))
    ax.cax.set_ylabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{o}}$")
    addWindow(ax)
    ret["pulls_obs"] = (fig, ax)

    fig, ax = plt.subplots(layout="tight")
    all_pulls = (pred_mean - raw_test.Y) / torch.sqrt(pred.V)
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
        just_window=True,
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
    base_dir,
    subpath,
    common_data=None,
    rebin=1,
    kernel=None,
    model_maker=None,
):
    common_data = common_data or {}

    torch.set_default_dtype(torch.float64)

    base_dir = Path(base_dir)
    save_dir = base_dir / subpath

    dirdat = futil.DirectoryData(base_dir)
    shutil.rmtree(save_dir, ignore_errors=True)
    save_dir.mkdir(parents=True, exist_ok=True)
    train_data = regression.makeRegressionData(
        inhist[hist.rebin(rebin), hist.rebin(rebin)], window_func, exclude_less=0.001
    )
    test_data = regression.makeRegressionData(
        inhist[hist.rebin(rebin), hist.rebin(rebin)], None, exclude_less=0.001
    )
    train_transform = regression.getNormalizationTransform(train_data)
    test_transform = regression.getNormalizationTransform(test_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = test_transform.transform(test_data)

    with subplots_context() as (fig, ax):
        out = save_dir / "original.pdf"
        plotting.drawAs2DHist(ax, plotting.PlotObject.fromHist(inhist))
        fig.savefig(out)
        dirdat.set(out, common_data)

    with subplots_context(1, 2, figsize=(10, 5)) as (fig, ax):
        out = save_dir / "masked_area.pdf"
        simpleGrid(ax[0], test_data.E, test_data.X, test_data.Y)
        simpleGrid(ax[1], train_data.E, train_data.X, train_data.Y)
        fig.savefig(out)
        dirdat.set(out, common_data)



    use_cuda = True 

    if torch.cuda.is_available() and use_cuda:   
        print("USING GPU")
        train=regression.sendToGpu(normalized_train_data)
    else:
        train = normalized_train_data

    model, likelihood = regression.createModel(
        train, kernel=kernel, model_maker=model_maker
    )
    if torch.cuda.is_available() and use_cuda:   
        model = model.cuda()
        likelihood = likelihood.cuda()
    # model,likelihood = regression.createModel(train, kernel=SK(models.GeneralRQ(ard_num_dims=2)))

    with linear_operator.settings.max_cg_iterations(2000):
        model, likelihood = regression.optimizeHyperparams(
            model, likelihood, train, bar=False, iterations=400
        )
    print("Done training")
    if torch.cuda.is_available() and use_cuda:   
        model = model.cpu()
        likelihood = likelihood.cpu()

    params = list(model.named_parameters_and_constraints())
    if False:
        for name, param, constraint in params:
            if constraint:
                real_param = constraint.transform(param)
            else:
                real_param = param
            print(
                f"{name.replace('raw_',''):{max(len(x[0]) for x in params)+4}} {real_param.detach().numpy().round(3)}"
            )


    with linear_operator.settings.max_cg_iterations(2000), gpytorch.settings.debug(
        False
    ):
        pred = regression.getPrediction(model, likelihood, normalized_test_data)

    pred = test_transform.iTransform(
        regression.DataValues(
            normalized_test_data.X, pred.mean, pred.variance, normalized_test_data.E
        )
    )
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
        print(f"Chi^2/bins = {chi2}")
        print(f"Avg Abs pull = {avg_pull}")
    else:
        mask = None
    diagnostic_plots = makeDiagnosticPlots(pred, test_data, train_data, inhist, mask)
    saveDiagnosticPlots(diagnostic_plots, save_dir, dirdat)
    makeSlicePlots(pred, test_data, inhist, window_func, 0, save_dir, dirdat)
    makeSlicePlots(pred, test_data, inhist, window_func, 1, save_dir, dirdat)

    p = save_dir / "metadata.yaml"
    torch.save(model.state_dict(), save_dir / "train_model.pth")
    with open(p, "w") as f:
        f.write(
            yaml.dump(
                {
                    "chi2_blinded": float(chi2),
                }
            )
        )


def scan(hist, window_func_generator):
    def mm(train_x, train_y, likelihood, kernel, **kwargs):
        return models.InducingPointModel(
            train_x, train_y, likelihood, kernel, inducing=train_x[::2]
        )

    NNRBF = models.wrapNN("NNRBFKernel", gpytorch.kernels.RBFKernel)
    nnrbf = SK(NNRBF(odim=4,layer_sizes=(32,16)))

    for name, wf in window_func_generator:
        path = name
        print(f"Now processing {name}")
        doCompleteRegression(
            hist,
            wf,
            "scan",
            path,
            model_maker=mm,
            kernel=nnrbf,
        )
        plt.close("all")


def main():
    from analyzer.core import AnalysisResult
    from analyzer.datasets import SampleManager

    res = AnalysisResult.fromFile("results/data_control.pkl")
    sample_manager = SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")
    bkg_name = "CR0b_Data2018"
    hists = res.getMergedHistograms(sample_manager)
    complete_hist = hists["ratio_m14_vs_m24"]
    orig = complete_hist[
        ..., hist.loc(1150) : hist.loc(3000), hist.loc(0.4) : hist.loc(1)
    ]
    narrowed = orig[..., :: hist.rebin(1), :: hist.rebin(1)]
    qcd_hist = narrowed[bkg_name, ...]
    qcd_hist = narrowed[bkg_name, ...]

    x_iter = map(float, torch.arange(1400, 2000, 250))
    x_size_iter = map(float, torch.arange(100, 400, 100))
    y_iter = map(float, torch.arange(0.5, 0.7, 0.15))
    y_size_iter = map(float, torch.arange(0.05, 0.15, 0.05))
    generator = (
        (
            f"E_{round(x)}_{round(y,2)}_{round(a)}_{round(b,2)}".replace(".", "p"),
            regression.ellipseMasker(torch.tensor([x, y]), a, b),
        )
        for x, a, y, b in it.product(x_iter, x_size_iter, y_iter, y_size_iter)
    )
    scan(qcd_hist, generator)


if __name__ == "__main__":
    loadStyles()
    main()
