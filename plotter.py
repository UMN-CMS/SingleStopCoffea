import hist
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
import pickle
import numpy as np
from pathlib import Path
import multiprocessing
from functools import partial
from rich.progress import track
import concurrent.futures


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

def draw1DHistogram(ax, edges, values, label, **kwargs):
    widths = np.diff(edges)
    ret = ax.bar(edges[:-1], values, width=widths,label=label, **kwargs )
    return ret


def make1DHistogram(ax, hist, samples):
    h = hist
    xax = next(x for x in h.axes if x.name != "dataset")
    sample_vals = [(s, h[s,:].to_numpy()) for s in samples if s in h.axes[0]]
    sample_vals = sorted(sample_vals, key=lambda a: np.sum(a[1][0]))
    bottom = np.zeros_like(sample_vals[0][1][0])
    for sample, (vals,edges) in sample_vals:
        draw1DHistogram(ax, edges, vals, sample,bottom=bottom)
        bottom += vals
    return ax


def drawScatter(ax, x,y,error=None, **kwargs):
    ret = ax.errorbar(x,y,yerr=error, **kwargs ,fmt='o')
    return ret

def makeScatter(ax, hists, samples):
    h = hists
    sample_vals = [(s, h[s,:].to_numpy()[1], h[s,:].view()) for s in samples if s in h.axes[0]]
    sample_vals = [(x,y[:-1] + np.diff(y)/2, z.value, z.variance) for x,y,z in sample_vals]
    sample_vals = sorted(sample_vals, key=lambda a: np.sum(a[2]))
    for sample, x,y,var in sample_vals:
        drawScatter(ax, x,y , error= var ,label=sample)
    return ax


def makefig(o, vals):
    n,h = vals
    if countHistAxes(h) == 2:
        f,ax = plt.subplots()
        ax = make1DHistogram(ax, h, ["TT2018", "Diboson2018",  "QCD2018", "ST2018", "WQQ2018", "ZQQ2018", "ZNuNu2018"])
        ax = makeScatter(ax, h, ['signal_2000_600_Skim', 'signal_1500_400_Skim', 'signal_2000_1900_Skim', 'signal_2000_400_Skim', 'signal_2000_900_Skim', 'signal_1500_600_Skim'])
        ax.set_yscale('log')
        xax = next(x for x in h.axes if x.name != "dataset")
        ax.set_xlabel(xax.label)
        ax.set_title(h.name)
        ax.set_ylabel("Weighted Events")
        ax.legend()
        addPrelim(ax)
        f.savefig(o / n)

if __name__ == '__main__':
    outdir = Path("plots")
    outdir.mkdir(parents=True, exist_ok=True)
    #makefig(outdir, ("m04_m", all_hists["m04_m"]))
    with concurrent.futures.ProcessPoolExecutor() as p:
        futs = p.map(partial(makefig, outdir), all_hists.items())
        for f in track(futs):
            pass
