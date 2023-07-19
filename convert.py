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
from analyzer.plotter import make2DSlicedProjection, autoPlot





def getHistograms(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r["histograms"]

all_hists = getHistograms("analyzer_output.pkl")



def gauss(x, *p):
    A, mu, sigma = p
    return A * np.exp(-((x - mu) ** 2) / (2.0 * sigma**2))

def cball(x, *p):
    m,b,loc,scale = p
    return crystalball.pdf(x,*p)


def makeOptimized(h_sig, h_bkg, opt_axis=None, disc_axis=None, spread=1):
    cur_max = None
    opt_axis = next(x for x in h_sig.axes if x.name == opt_axis)
    disc_axis = next(x for x in h_sig.axes if x.name == disc_axis)
    cur_best = 0

    hx = h_sig[{disc_axis.name: sum}]
    vals, edges = hx.to_numpy()
    edges = (edges[:-1] + edges[1:]) / 2
    p0 = [1000, 0.9, 0.05]
    coeff, var_matrix = curve_fit(gauss, edges, vals, p0=p0, bounds=[(0,0,0), (1e10,1e10,0.15)])
    _, mu, sig = coeff
    cutupper = mu + spread * sig
    cutlower = mu - spread * sig
    print(coeff)

    #coeff, var_matrix = curve_fit(cball, edges, vals, p0=p0, bounds=[(0,0,0), (1e10,1e10,1e10)])
    #m,b,loc,scale = coeff
    #cutupper = 
    #cutlower = mu - spread * sig
    #print(coeff)

    s = hist.tag.Slicer()

    return (
        h_sig[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
        h_bkg[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
        coeff,
    )

def makeOptimized(h_sig, h_bkg, opt_axis=None, disc_axis=None, width_count=10,start_count=10, width_min=0.1, width_max=0.4, start_min=0.1, start_max=0.8):
    cur_max = None
    opt_axis = next(x for x in h_sig.axes if x.name == opt_axis)
    disc_axis = next(x for x in h_sig.axes if x.name == disc_axis)
    cur_best = 0
    low,high = opt_axis[0], opt_axis[-1]
    min_bin_edge, max_bin_edge = low[0], high[1]
    diff = max_bin_edge-min_bin_edge
    widths = np.linspace(diff * width_min, diff*width_max, width_count)
    starts =  np.linspace(min_bin_edge + start_min * diff, min_bin_edge + start_max * diff, start_count)
    cur_max = 0
    best_start, best_width = 0, 0
    s = hist.tag.Slicer()
    for width in widths:
        these_starts = starts[(starts-width) > 0]
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


#def makeOptimized(h_sig, h_bkg, opt_axis=None, disc_axis=None):
#    cur_max = None
#    opt_axis = next(x for x in h_sig.axes if x.name == opt_axis)
#    disc_axis = next(x for x in h_sig.axes if x.name == disc_axis)
#    cur_best = 0
#
#    hx = h_sig[{disc_axis.name: sum}]
#    vals, edges = hx.to_numpy()
#    edges = (edges[:-1] + edges[1:]) / 2
#    p0 = [50, 1000, 50]
#    coeff, var_matrix = curve_fit(gauss, edges, vals, p0=p0)
#    _, mu, sig = coeff
#    cutupper = mu + 2.5 * sig
#    cutlower = mu - 2.5 * sig
#    s = hist.tag.Slicer()
#
#    return (
#        h_sig[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
#        h_bkg[{opt_axis.name: s[hist.loc(cutlower) : hist.loc(cutupper) : sum]}],
#        coeff,
#    )




signals = [x for x in all_hists["HT"].axes[0] if "signal" in x]
outdir = Path("ratio_out")
mapping = {}

for sig in signals:
    print(sig)
    m = re.match(r"signal_312_(\d+)_(\d+)", sig)
    m1, m2 = m.group(1, 2)
    m1, m2 = int(m1), int(m2)
    if abs(m1 - m2) > 200:
        target = "m14_vs_m13"
        target = "ratio_m14_vs_m3_top_3_no_lead_b"
        a1 = "13"
        a1 = "ratio"
    else:
        target = "m14_vs_m24"
        target = "ratio_comp_m14_vs_m24"
        target = "ratio_m14_vs_m3_top_2_plus_lead_b"
        a1 = "ratio"
        a1 = "ratio"
    hi = all_hists[target]
    hs = hi[sig, sum, ...]
    hb = hi[["QCDInclusive2018"], ...,]
    hb = hb[sum, sum, ...]
    print(hb.axes[0][0])
    print(hb.axes[0][-1])

    s1 = hs.sum().value
    s2 = hb.sum().value
    print("#" * 50)
    print(f"Signal: {sig}")
    print(f"Signal sum is {s1}")
    print(f"Bkg sum is {s2}")


    print(hs)
    spread = 1
    #bs, bb, (A,mu,sigma) = makeOptimized(hs, hb, a1, "m14", spread)
    bs, bb, window = makeOptimized(hs, hb, a1, "m14")
    #print(f"Coefficients are sigma = {sigma} mu = {mu}")
    #window = (mu - spread * sigma , mu + spread*sigma)
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
    autoPlot(
        outdir / sig / "sig.pdf",
        make2DSlicedProjection,
        hs,
        bs,
        #add_fit=lambda x: gauss(x, A, mu , sigma),
        hlines=[cutlower, cutupper],
        #fig_title=sig,
        #fig_manip=lambda f: f.text(
        #    0.9,
        #    0.9,
        #    f"{sig}",
        #    horizontalalignment="right",
        #    verticalalignment="top",
        #)
    )
    autoPlot(
        outdir / sig / "bkg.pdf",
        make2DSlicedProjection,
        hb,
        bb,
        add_fit=None,
        hlines=[cutlower, cutupper],
        #fig_params=dict(figsize=(12, 10)),
        #fig_title="QCD",
        #fig_manip=lambda f: f.text(
        #    0.9,
        #    0.9,
        #    "Backgrounds",
        #    horizontalalignment="right",
        #    verticalalignment="top",
        #)
    )
    data = dict(
        before_sig=before_sig,
        after_sig=after_sig,
        #coeffs=dict(mu=mu, sigma=sigma)
    )
    with open(outdir / sig / "data.json", "w") as f:
        f.write(json.dumps(data, indent=4))

    root_output = uproot.recreate(outdir / sig / "hists.root")
    root_output[f"{sig}_{target}_opt_{a1}_proj_04"] = bs
    root_output[f"bkg_{sig}_{target}_opt_{a1}_proj_04"] = bb

    root_output[f"{sig}_{target}"] = hs[sum,...]
    root_output[f"bkg_{sig}_{target}"] = hb[sum,...]

    ms,mx=int(m1), int(m2)

    mapping[sig] = dict(
        stop_mass = int(ms), 
        chi_mass = int(mx),
        base_dir=f"signal_312_{ms}_{mx}",
        hists=str("hists.root"),
        bkg_hist_name=f"bkg_{sig}_{target}_opt_{a1}_proj_04",
        sig_hist_name=f"{sig}_{target}_opt_{a1}_proj_04",
        base_bkg_hist_name= f"bkg_{sig}_{target}",
        base_sig_hist_name= f"{sig}_{target}"
    )

outdir.mkdir(exist_ok=True)
mfile = outdir / "all_data.json"
with open(mfile, "w") as f:
    f.write(json.dumps(mapping))
