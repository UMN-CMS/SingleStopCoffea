from pathlib import Path

import numpy as np

import gpytorch
import linear_operator
import torch
import uproot
from fitting.high_level import RegressionModel, SignalData
from fitting.utils import fixMVN, getScaledEigenvecs

from .datacard import Channel, DataCard, Process, Systematic


def tensorToHist(array):
    a = array.numpy()
    hist = (a, np.arange(0, a.shape[0] + 1))
    return hist


def createHists(regression_data, signal_data, root_file, num_bkg_systs=None):
    bm = regression_data.domain_mask
    cov_mat = regression_data.posterior_dist.covariance_matrix
    mean = regression_data.posterior_dist.mean
    ev = getScaledEigenvecs(cov_mat, top=num_bkg_systs).T

    root_file["bkg_estimate"] = tensorToHist(mean)
    root_file["signal"] = tensorToHist(signal_data.Y[bm])

    root_file["data_obs"] = tensorToHist(regression_data.test_data.Y + 0 * signal_data.Y[bm])

    for i, v in enumerate(ev):
        #print(f"Magnitude is {torch.abs(v).max()}")
        h_up = tensorToHist(mean + v)
        root_file[f"bkg_estimate_EVAR_{i}Up"] = h_up
        h_down = tensorToHist(mean - v)
        root_file[f"bkg_estimate_EVAR_{i}Down"] = h_down


def createDatacard(regression_data, signal_data, output_dir, num_bkg_systs=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    root_path = output_dir / "histograms.root"
    root_file = uproot.recreate(root_path)

    if num_bkg_systs is None:
        num_bkg_systs = regression_data.test_data.Y.shape[0]
    createHists(regression_data, signal_data, root_file, num_bkg_systs)

    card = DataCard()

    bkg = Process("BackgroundEstimate", False)
    sig = Process("Signal", True)
    b1 = Channel("SignalRegion")
    card.addChannel(b1)
    card.addProcess(sig)
    card.addProcess(bkg)

    card.setProcessRate(sig, b1, -1)
    card.setProcessRate(bkg, b1, -1)

    card.addShape(
        bkg,
        b1,
        "histograms.root",
        "bkg_estimate",
        "bkg_estimate_$SYSTEMATIC",
    )
    card.addShape(sig, b1, "histograms.root", "signal", "")

    card.addObservation(
        b1, "histograms.root", "data_obs", int(torch.sum(regression_data.test_data.Y))
    )


    for i in range(0, num_bkg_systs):
        s = Systematic(f"EVAR_{i}", "shape")
        card.addSystematic(s)
        card.setProcessSystematic(bkg, s, b1, 1)

    with open(output_dir / "datacard.txt", "w") as f:
        f.write(card.dumps())


def main():
    print("Starting")
    import sys

    path = Path(sys.argv[1]).absolute()
    spath = Path(sys.argv[2]).absolute()
    d = torch.load(path)
    s = torch.load(spath)
    pd = d.posterior_dist
    sd = s.signal_data

    createDatacard(d, sd, "combineoutput/testout", 300) #int(pd.mean.size(0)))


if __name__ == "__main__":
    main()
