import pickle
import uproot
import hist
import math
import itertools as it
import re
import numpy as np
from pathlib import Path
import json
from scipy.optimize import curve_fit
from plotter import make2DSlicedProjection, autoPlot


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def getHistograms(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r


all_hists = getHistograms("output.pkl")


def makeOptimized(h_sig, h_bkg, opt_axis=None, disc_axis=None):
    cur_max = None
    opt_axis = next(x for x in h_sig.axes if x.name == opt_axis)
    disc_axis = next(x for x in h_sig.axes if x.name == disc_axis)
    cur_best = 0

    hx = h_sig[{disc_axis.name: sum}]
    vals, edges = hx.to_numpy()
    edges = (edges[:-1] + edges[1:]) / 2
    p0 = [50, 1000, 50]
    coeff, var_matrix = curve_fit(gauss, edges, vals, p0=p0)
    _, mu, sig = coeff
    cutupper = mu + 2.5 * sig
    cutlower = mu - 2.5 * sig
    s = hist.tag.Slicer()

    return (
        h_sig[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
        h_bkg[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
        coeff,
    )


def makeOptimized(h_sig, h_bkg, opt_axis=None, disc_axis=None):
    cur_max = None
    opt_axis = next(x for x in h_sig.axes if x.name == opt_axis)
    disc_axis = next(x for x in h_sig.axes if x.name == disc_axis)
    cur_best = 0

    width_start, width_stop, width_step = 200, 600, 100
    start, stop, step = 100, 2000, 50
    cur_max = 0
    best_start, best_width = 0, 0
    s = hist.tag.Slicer()
    for start, width in it.product(
        range(start, stop, step), range(width_start, width_stop, width_step)
    ):
        sum_sig = h_sig[
            {
                opt_axis.name: s[hist.loc(start) : hist.loc(start + width) : sum],
                disc_axis.name: sum,
            }
        ]
        sum_bkg = h_bkg[
            {
                opt_axis.name: s[hist.loc(start) : hist.loc(start + width) : sum],
                disc_axis.name: sum,
            }
        ]
        sig = sum_sig.value / np.sqrt(sum_bkg.value)
        if cur_max < sig:
            cur_max = sig
            best_start, best_width = start, width
    return (
        h_sig[
            {
                opt_axis.name: s[
                    hist.loc(best_start) : hist.loc(best_start + best_width) : sum
                ]
            }
        ],
        h_bkg[
            {
                opt_axis.name: s[
                    hist.loc(best_start) : hist.loc(best_start + best_width) : sum
                ]
            }
        ],
        (best_start, best_width),
    )


signals = [x for x in all_hists["HT"].axes[0] if "signal" in x]
outdir = Path("plots")
mapping = {}

for sig in signals:
    m = re.match(r"signal_(\d+)_(\d+)_Skim", sig)
    m1, m2 = m.group(1, 2)
    m1, m2 = int(m1), int(m2)
    if abs(m1 - m2) <= 200:
        target = "m03_vs_m04"
        a1 = "03"
    else:
        target = "m14_vs_m04"
        a1 = "14"
    hi = all_hists[target]
    hs = hi[sig, sum, ...]
    hb = hi[
        [
            "QCD2018",
            "TT2018",
            "Diboson2018",
            "WQQ2018",
            "ZQQ2018",
            "ZNuNu2018",
            "ST2018",
        ],
        ...,
    ]
    hb = hb[sum, sum, ...]

    bs, bb, window = makeOptimized(hs, hb, f"mass_{a1}", "mass_04")

    before_sig = hs.sum().value / np.sqrt(hb.sum().value)
    after_sig = bs.sum().value / np.sqrt(bb.sum().value)

    print(f"Before: {before_sig}     After: {after_sig}")
    # _, mu, sigma = fit
    # cutupper = mu + 2.5 * sigma
    # cutlower = mu - 2.5 * sigma
    cutlower = window[0]
    cutupper = window[0] + window[1]
    print(f"Found window {cutlower} -  {cutupper} ")
    autoPlot(
        outdir / sig / "sig.pdf",
        make2DSlicedProjection,
        hs,
        bs,
        add_fit=None,  # lambda x: gauss(x, *fit),
        vlines=[cutlower, cutupper],
        fig_params=dict(figsize=(12, 10)),
        fig_manip=lambda f: f.text(
            0.9,
            0.9,
            f"{sig}",
            horizontalalignment="right",
            verticalalignment="top",
        )
    )
    autoPlot(
        outdir / sig / "bkg.pdf",
        make2DSlicedProjection,
        hb,
        bb,
        add_fit=None,
        vlines=[cutlower, cutupper],
        fig_params=dict(figsize=(12, 10)),
        fig_manip=lambda f: f.text(
            0.9,
            0.9,
            "Backgrounds",
            horizontalalignment="right",
            verticalalignment="top",
        )
    )
    data = dict(
        before_sig=before_sig,
        after_sig=after_sig,
        # coeffs=dict(mu=mu, sigma=sigma)
    )
    with open(outdir / sig / "data.json", "w") as f:
        f.write(json.dumps(data, indent=4))

    root_output = uproot.recreate(outdir / sig / "hists.root")
    root_output[f"{sig}_{target}_opt_{a1}_proj_04"] = bs
    root_output[f"QCD2018_{sig}_{target}_opt_{a1}_proj_04"] = bb

    mapping[sig] = dict(
        base_dir=str(sig),
        hists=str("hists.root"),
        bkg_hist_name=f"QCD2018_{sig}_{target}_opt_{a1}_proj_04",
        sig_hist_name=f"{sig}_{target}_opt_{a1}_proj_04",
    )
    break

mfile = outdir / "all_data.json"
with open(mfile, "w") as f:
    f.write(json.dumps(mapping))
