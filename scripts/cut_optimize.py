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
from scipy.stats import crystalball
import sys

sys.path.append(".")
from analyzer.datasets import loadSamplesFromDirectory
from analyzer.plotting.core_plots import *

loadStyles()

lumi = 137.6

manager = loadSamplesFromDirectory("datasets")


def getHistograms(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r["histograms"]


all_hists = getHistograms("signalhists.pkl")


def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))


def cball(x, *p):
    m, b, loc, scale = p
    return crystalball.pdf(x, *p)


def makeOptimized(h_sig, h_bkg, opt_axis=None, disc_axis=None, spread=1):
    cur_max = None
    opt_axis = next(x for x in h_sig.axes if x.name == opt_axis)
    disc_axis = next(x for x in h_sig.axes if x.name == disc_axis)
    cur_best = 0

    hx = h_sig[{disc_axis.name: sum}]
    vals, edges = hx.to_numpy()
    edges = (edges[:-1] + edges[1:]) / 2
    p0 = [1000, 0.9, 0.05]
    coeff, var_matrix = curve_fit(
        gauss, edges, vals, p0=p0, bounds=[(0, 0, 0), (1e10, 1e10, 0.15)]
    )
    _, mu, sig = coeff
    cutupper = mu + spread * sig
    cutlower = mu - spread * sig
    print(coeff)

    # coeff, var_matrix = curve_fit(cball, edges, vals, p0=p0, bounds=[(0,0,0), (1e10,1e10,1e10)])
    # m,b,loc,scale = coeff
    # cutupper =
    # cutlower = mu - spread * sig
    # print(coeff)

    s = hist.tag.Slicer()

    return (
        h_sig[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
        h_bkg[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
        coeff,
    )


def makeOptimized(
    h_sig,
    h_bkg,
    opt_axis=None,
    disc_axis=None,
    width_count=10,
    start_count=10,
    width_min=0.1,
    width_max=0.4,
    start_min=0.1,
    start_max=0.8,
):
    cur_max = None
    opt_axis = h_sig.axes[opt_axis]
    disc_axis = h_sig.axes[disc_axis]
    cur_best = 0
    low, high = opt_axis[0], opt_axis[-1]
    min_bin_edge, max_bin_edge = low[0], high[1]
    diff = max_bin_edge - min_bin_edge
    widths = np.linspace(diff * width_min, diff * width_max, width_count)
    starts = np.linspace(
        min_bin_edge + start_min * diff, min_bin_edge + start_max * diff, start_count
    )
    cur_max = 0
    best_start, best_width = 0, 0
    s = hist.tag.Slicer()
    for width in widths:
        these_starts = starts[(starts - width) > 0]
        for start in these_starts:
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
            sig = sum_sig.value / np.sqrt(sum_bkg.value) * np.sqrt(sum_sig.value)
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
        (best_start, best_start + best_width),
    )


signals = [x for x in all_hists["HT"].axes[0] if "signal" in x]
outdir = Path("cutoptimized")
outdir.mkdir(exist_ok=True)
mapping = {}

for sig in signals:
    print(sig)
    (outdir / sig).mkdir(exist_ok=True)
    m = re.match(r"signal_312_(\d+)_(\d+)", sig)
    m1, m2 = m.group(1, 2)
    m1, m2 = int(m1), int(m2)
    if abs(m1 - m2) > 200:
        label = "Uncompressed"
        target = "m14_vs_m13"
        target = "ratio_m14_vs_m3_top_3_no_lead_b"
        # target = "ratio_m14_vs_m13"
        a1 = "13"
        a1 = "ratio"
        a1 = 1
    else:
        label = "Compressed"
        target = "m14_vs_m24"
        target = "ratio_comp_m14_vs_m24"
        target = "ratio_m14_vs_m13"  # _top_2_plus_lead_b"
        a1 = "ratio"
        a1 = "ratio"
        a1 = 1
    hi = all_hists[target]
    hs = hi[sig, ...]
    hb = hi[
        "Skim_QCDInclusive2018",
        ...,
    ]
    s1 = hs.sum().value
    s2 = hb.sum().value
    print("#" * 50)
    print(f"Signal: {sig}")
    print(f"Signal sum is {s1}")
    print(f"Bkg sum is {s2}")
    spread = 1
    # bs, bb, (A,mu,sigma) = makeOptimized(hs, hb, a1, "m14", spread)
    bs, bb, window = makeOptimized(hs, hb, a1, 0)
    # print(f"Coefficients are sigma = {sigma} mu = {mu}")
    # window = (mu - spread * sigma , mu + spread*sigma)
    print(f"Window is  ")
    before_sig = hs.sum().value / np.sqrt(hb.sum().value)
    after_sig = bs.sum().value / np.sqrt(bb.sum().value)

    print(f"Before: {before_sig}     After: {after_sig}")
    # _, mu, sigma = fit
    # cutupper = mu + 2.5 * sigma
    # cutlower = mu - 2.5 * sigma
    cutlower = window[0]
    cutupper = window[1]
    print(f"Found window {cutlower} -  {cutupper} ")

    #############################################################
    fig, ax = drawAs2DExtended(
        PlotObject(hs, sig, manager[sig]),
        top_stack=[
            PlotObject(hs[:, sum], sig, manager[sig]),
            PlotObject(bs, sig, manager[sig]),
        ],
        right_stack=[
            PlotObject(hs[sum, :], sig, manager[sig]),
        ],
        top_pad=0.2,
    )
    for a, l in it.product([ax, *ax.right_axes], window):
        a.axhline(y=l)
    addPrelim(
        ax,
        additional_text="\n$\\lambda_{312}''$"
        + f" {label}\n"
        + f"{manager[sig].getTitle()}",
        pos="in",
        color="white",
    )
    addEra(ax.top_axes[-1], lumi or 59.8)
    ax = addTitles2D(ax, hs)

    fig.savefig(outdir / sig / "sig.pdf")
    plt.close(fig)

    #############################################################
    bkg = "Skim_QCDInclusive2018"
    fig, ax = drawAs2DExtended(
        PlotObject(hb, bkg, manager[bkg]),
        top_stack=[
            PlotObject(hb[:, sum], bkg, manager[bkg]),
            PlotObject(bb, bkg, manager[bkg]),
        ],
        right_stack=[
            PlotObject(hb[sum, :], bkg, manager[bkg]),
        ],
        # top_pad=0.2
    )
    addTitles2D(ax, hb)
    for a, l in it.product([ax, *ax.right_axes], window):
        a.axhline(y=l)
    addPrelim(
        ax,
        additional_text="\n$\\lambda_{312}''$"
        + f" {label}\n"
        + f"{manager[bkg].getTitle()}",
        pos="in",
        color="white",
    )
    addEra(ax.top_axes[-1], lumi or 59.8)
    for a in ax.top_axes:
        ot = a.get_yaxis().get_offset_text()
        if not ot:
            continue
        x,y = ot.get_position()
        ot.set_position((x-0.15,y))
    fig.savefig(outdir / sig / "bkg.pdf")
    plt.close(fig)
    data = dict(
        before_sig=before_sig,
        after_sig=after_sig,
        # coeffs=dict(mu=mu, sigma=sigma)
    )
    with open(outdir / sig / "data.json", "w") as f:
        f.write(json.dumps(data, indent=4))

    root_output = uproot.recreate(outdir / sig / "hists.root")
    root_output[f"{sig}_{target}_opt_{a1}_proj_04"] = bs
    root_output[f"bkg_{sig}_{target}_opt_{a1}_proj_04"] = bb

    root_output[f"{sig}_{target}"] = hs[sum, ...]
    root_output[f"bkg_{sig}_{target}"] = hb[sum, ...]

    ms, mx = int(m1), int(m2)

    mapping[sig] = dict(
        stop_mass=int(ms),
        chi_mass=int(mx),
        base_dir=f"signal_312_{ms}_{mx}",
        hists=str("hists.root"),
        bkg_hist_name=f"bkg_{sig}_{target}_opt_{a1}_proj_04",
        sig_hist_name=f"{sig}_{target}_opt_{a1}_proj_04",
        base_bkg_hist_name=f"bkg_{sig}_{target}",
        base_sig_hist_name=f"{sig}_{target}",
    )

mfile = outdir / "all_data.json"
with open(mfile, "w") as f:
    f.write(json.dumps(mapping))
