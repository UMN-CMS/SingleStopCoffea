import hist
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import pickle
import numpy as np
from pathlib import Path
import multiprocessing
from functools import partial


font_dirs = ['./fonts' ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
for font in font_files:
    font_manager.fontManager.addfont(font)

plt.style.use("style.mplstyle")


def getHistograms(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r


all_hists = getHistograms("output.pkl")


def draw1DHistogram(ax, edges, values, label, **kwargs):
    widths = np.diff(edges)
    ret = ax.bar(edges[:-1], values, width=widths,label=label, **kwargs )
    return ret


def addPrelim(ax):
    ax.text(
        0.02,
        0.98,
        "CMS Preliminary",
        style="italic",
        fontweight="bold",
        horizontalalignment="left",
        verticalalignment="top",
        transform = ax.transAxes
    )


def countHistAxes(hist):
    return len(hist.axes)

def make1DHistogram(hist, samples, name):
    fig, ax = plt.subplots()
    addPrelim(ax)
    h = hist
    xax = next(x for x in h.axes if x.name != "dataset")
    sample_vals = [(s, h[s,:].to_numpy()) for s in samples if s in h.axes[0]]
    sample_vals = sorted(sample_vals, key=lambda a: np.sum(a[1][0]))
    #summed = np.cumsum([v[1][0] for v in sample_vals],axis=0)
    #stacked_vals = list((x[0], (*x[1], y)) for x,y in  zip(sample_vals, summed))
    bottom = np.zeros_like(sample_vals[0][1][0])
    for sample, (vals,edges) in sample_vals:
        draw1DHistogram(ax, edges, vals, sample,bottom=bottom)
        bottom += vals

    ax.set_yscale('log')
    ax.set_xlabel(xax.label)
    ax.set_title(hist.name)
    ax.set_ylabel("Weighted Events")
    ax.legend()
    return fig






def makefig(o, vals):
    n,h = vals
    if countHistAxes(h) == 2:
        f = make1DHistogram(h, ["TT2018", "Diboson2018",  "QCD2018", "ST2018", "WQQ2018", "ZQQ2018", "ZNuNu2018"], n)
        f.savefig(o / n)

if __name__ == '__main__':
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)
    p = multiprocessing.Pool(8)
    p.map(partial(makefig, outdir), all_hists.items())
