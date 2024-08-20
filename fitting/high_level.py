import itertools as it
import random
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

import analyzer.file_utils as futil
import analyzer.plotting as plotting
import fitting.utils as fit_utils
import gpytorch
import hist
import linear_operator
import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
import uhi
from analyzer.plotting.mplstyles import loadStyles
from gpytorch.kernels import ScaleKernel as SK
from matplotlib.patches import Polygon

from . import models, regression, transformations, windowing
from .plot_tools import createSlices, getPolyFromSquares, makeSquares, simpleGrid

torch.set_default_dtype(torch.float64)


@dataclass
class RegressionModel:
    input_data: hist.Hist
    window: Any
    domain_mask: torch.Tensor

    train_data: torch.Tensor
    test_data: torch.Tensor

    trained_model: gpytorch.models.ExactGP

    raw_posterior_dist: gpytorch.distributions.MultivariateNormal
    posterior_dist: gpytorch.distributions.MultivariateNormal


@dataclass
class SignalData:
    signal_data: torch.Tensor
    domain_mask: torch.Tensor = None
    signal_hist: hist.Hist = None
    signal_name: str = None


def saveDiagnosticPlots(plots, dirdata, save_dir):
    for name, (fig, ax) in plots.items():
        o = (save_dir / name).with_suffix(".pdf")
        fig.savefig(o)


def makeSigBkgPlot(train_data, test_data, signal_data, window):
    fig, ax = plt.subplots(1, 2, figsize=(10, 5), layout="tight")
    mask = regression.getBlindedMask(
        test_data.X, test_data.Y, test_data.Y, test_data.V, window
    )
    squares = makeSquares(test_data.X[mask], test_data.E)
    points = getPolyFromSquares(squares)
    # plotting.drawAs2DHist(ax[0][1], plotting.PlotObject.fromHist(sig_hist))

    simpleGrid(ax[0], train_data.E, train_data.X, train_data.Y)
    simpleGrid(ax[1], signal_data.E, signal_data.X, signal_data.Y)
    poly = mpl.patches.Polygon(points, edgecolor="red", fill=False)
    ax[0].add_patch(poly)
    poly = mpl.patches.Polygon(points, edgecolor="red", fill=False)
    ax[1].add_patch(poly)
    return {"sig_bkg": (fig, ax)}


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
    p = plotting.PlotObject.fromNumpy(
        np.histogram(all_pulls[torch.abs(all_pulls) < np.inf], bins=20)
    )
    print("HERE")
    plotting.drawAs1DHist(ax, p, yerr=False)
    ax.set_xlabel(r"$\frac{N_{obs}-N_{pred}}{\sigma_{p}}$")
    ax.set_ylabel("Count")
    ret["pulls_hist"] = (fig, ax)

    return ret


def makeSlicePlots(pred, test_data, hist, window, dim, save_dir, dirdata):
    pred_mean, _ = fit_utils.pointsToGrid(test_data.X, pred.Y, test_data.E)
    pred_var, _ = fit_utils.pointsToGrid(test_data.X, pred.V, test_data.E)
    obs_vals, _ = fit_utils.pointsToGrid(test_data.X, test_data.Y, test_data.E)
    obs_vars, filled = fit_utils.pointsToGrid(test_data.X, test_data.V, test_data.E)

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
    test_data, domain_mask = regression.makeRegressionData(
        inhist[hist.rebin(rebin), hist.rebin(rebin)],
        None,
        exclude_less=0.001,
        get_mask=True,
    )
    train_transform = transformations.getNormalizationTransform(train_data)
    normalized_train_data = train_transform.transform(train_data)
    normalized_test_data = train_transform.transform(test_data)

    use_cuda = True

    if torch.cuda.is_available() and use_cuda:
        print("USING GPU")
        train = normalized_train_data.toGpu()
        norm_test = normalized_test_data.toGpu()
    else:
        train = normalized_train_data
        norm_test = normalized_test_data

    lr = 0.05
    ok = False
    while not ok:
        model, likelihood = regression.createModel(
            train, kernel=kernel, model_maker=model_maker, learn_noise=False
        )

        if hasattr(model.covar_module, "initialize_from_data"):
            model.covar_module.initialize_from_data(train.X, train.Y)
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
                lr=lr,
                get_evidence=True,
            )
            if torch.cuda.is_available() and use_cuda:
                model = model.cpu()
                likelihood = likelihood.cpu()
            print("Done with loop")
            raw_pred_dist = regression.getPrediction(
                model, likelihood, normalized_test_data
            )
            with gpytorch.settings.cholesky_max_tries(30):
                psd_pred_dist = fit_utils.fixMVN(raw_pred_dist)
            raw_pred_dist = type(raw_pred_dist)(
                psd_pred_dist.mean, psd_pred_dist.covariance_matrix.to_dense()
            )
            ok = True
        except (
            linear_operator.utils.errors.NanError,
            linear_operator.utils.errors.NotPSDError,
        ) as e:
            lr = lr + random.random() / 1000
            print(f"CHOLESKY FAILED: retrying with lr={round(lr,3)}")
            print(e)

    print("Done training")

    slope = train_transform.transform_y.slope
    intercept = train_transform.transform_y.intercept
    pred_dist = fit_utils.affineTransformMVN(psd_pred_dist, slope, intercept)

    pred_data = regression.DataValues(
        test_data.X,
        pred_dist.mean,
        pred_dist.variance,
        test_data.E,
    )

    data = {
        "evidence": evidence,
        "model_string": str(model),
    }

    if window_func:
        mask = regression.getBlindedMask(
            pred_data.X, pred_data.Y, test_data.Y, test_data.V, window_func
        )
        bpred_mean = pred_data.Y[mask]
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
    if True:
        diagnostic_plots = makeDiagnosticPlots(
            pred_data, test_data, train_data, inhist, mask
        )
        saveDiagnosticPlots(diagnostic_plots, dir_data, save_dir)
        makeSlicePlots(pred_data, test_data, inhist, window_func, 0, save_dir, dir_data)
        makeSlicePlots(pred_data, test_data, inhist, window_func, 1, save_dir, dir_data)

    save_data = RegressionModel(
        input_data=inhist,
        window=window_func,
        domain_mask=domain_mask,
        train_data=train_data,
        test_data=test_data,
        trained_model=model,
        raw_posterior_dist=raw_pred_dist,
        posterior_dist=pred_dist,
    )
    torch.save(save_data, save_dir / "train_model.pth")
    dir_data.setGlobal(data)
    return save_data


def createWindowForSignal(signal_data, axes=(150, 0.05)):
    max_idx = torch.argmax(signal_data.Y)
    max_x = signal_data.X[max_idx].round(decimals=2)
    return windowing.EllipseWindow(max_x.tolist(), list(axes))


def doEstimationForSignals(signals, bkg_hist, kernel, base_dir, kernel_name=""):
    base_dir = Path(base_dir)
    inducing_ratio = 1

    def mm(train_x, train_y, likelihood, kernel, **kwargs):
        return models.InducingPointModel(
            train_x, train_y, likelihood, kernel, inducing=train_x[::inducing_ratio]
        )

    i = 0
    for signal_name, signal_hist in signals:
        i = i + 1

        print("=" * 20)
        print(f"Signal is: {signal_name}")
        if kernel_name:
            print(f"KERNEL: {kernel_name}")

        if signal_hist:
            signal_regression_data = regression.makeRegressionData(signal_hist)
            window = createWindowForSignal(signal_regression_data)
            sd = SignalData(signal_regression_data, None, signal_hist, signal_name)
        else:
            window = None

        print(f"Now processing area {window.toString() if window else 'NoWindow'}")

        path = base_dir / (window.toString() if window else "NoWindow")
        shutil.rmtree(path, ignore_errors=True)
        path.mkdir(parents=True, exist_ok=True)

        if signal_hist:
            torch.save(sd, path / "signal_data.pth")

        dirdata = futil.DirectoryData(path)
        dir_data = {"inducing_ratio": inducing_ratio, "signal": signal_name or "None"}
        if window:
            dir_data["window"] = window.toDict()
        dirdata.setGlobal(dir_data)

        d = doCompleteRegression(
            bkg_hist, window, dirdata, model_maker=mm, kernel=kernel
        )

        if signal_hist:
            r = makeSigBkgPlot(
                d.train_data, d.test_data, signal_regression_data, window
            )
            saveDiagnosticPlots(r, dirdata, dirdata.directory)

        plt.close("all")
        print("=" * 20)


def main():
    from analyzer.datasets import SampleManager, ProfileRepo
    from analyzer.core import AnalysisResult

    mpl.use("Agg")

    res = AnalysisResult.fromFile("results/histograms/QCDInclusive2018.pkl")
    sig = AnalysisResult.fromFile("results/histograms/signal.pkl")

    profile_repo = ProfileRepo()
    profile_repo.loadFromDirectory("profiles")
    sample_manager = SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets", profile_repo, False)

    hists = res.getMergedHistograms(sample_manager)
    sig_hists = sig.getMergedHistograms(sample_manager)

    complete_hist = hists["ratio_m14_vs_m24"]["QCDInclusive2018"]
    signal_hists_complete = sig_hists["ratio_m14_vs_m24"]
    orig = complete_hist[
        ..., hist.loc(1150) : hist.loc(3000), hist.loc(0.4) : hist.loc(1)
    ]
    narrowed = orig[..., :: hist.rebin(2), :: hist.rebin(2)]
    qcd_hist = narrowed
    # qcd_hist = narrowed[bkg_name, ...] * 0.09764933859427383

    signal_hist_names = [
        "signal_312_1500_600",
        "signal_312_2000_1400",
        "signal_312_1500_900",
        "signal_312_2000_900",
    ]
    signals_to_scan = [(sn, signal_hists_complete[sn]) for sn in signal_hist_names]
    signals_to_scan.append((None, None))

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

    shape = (256, 128, 64, 32, 16)
    nnrbf_deep = SK(models.NNRBFKernel(odim=2, layer_sizes=shape))
    nnrbf_huge = SK(models.NNRBFKernel(odim=2, layer_sizes=(1000, 500, 10)))

    nngrbf = SK(models.NNGRBFKernel(odim=2, layer_sizes=(32, 16, 4)))

    nnsmk_tiny = models.NNSMKernel(odim=2, layer_sizes=(500, 250, 50), num_mixtures=4)

    grq = SK(models.GeneralRQ(ard_num_dims=2))

    rbf = SK(gpytorch.kernels.RBFKernel(ard_num_dims=2))

    grbf = SK(models.GeneralRBF(ard_num_dims=2))
    gsmk = models.GeneralSpectralMixture(ard_num_dims=2, num_mixtures=4)

    smk = gpytorch.kernels.SpectralMixtureKernel(num_mixtures=12, ard_num_dims=2)

    kernels = {
        # "rbf": rbf,
        # "smk": smk,
        # "grq": grq,
        # "nngrf" : nngrbf,
        # "nnsmk_tiny": nnsmk_tiny,
        # "nnrbf_deep": nnrbf_deep,
        # "nnrbf_huge": nnrbf_huge,
        # "nnrbf_tiny": nnrbf_tiny,
        # "nnrbf_32_16_8": nnrbf32_16_8,
        "nnrbf_32_32_8": SK(models.NNRBFKernel(odim=2, layer_sizes=(32, 32, 8))),
    }

    p = Path("allscans")
    for n, k in kernels.items():
        doEstimationForSignals(signals_to_scan, qcd_hist, k, p / n, kernel_name=n)


if __name__ == "__main__":
    loadStyles()
    main()
