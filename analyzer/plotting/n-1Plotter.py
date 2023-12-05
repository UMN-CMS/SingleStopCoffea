import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import importlib.resources as imp_res
import static
from dataclasses import dataclass
from functools import partial, wraps
import itertools as it
import hist
import re
import pickle
from analyzer.plotting.core_plots import *

def drawAs1DHist(ax, plot_object, yerr=True):
    h = plot_object.hist
    a = h.axes[0]
    edges = a.edges
    widths = a.widths
    vals = h.values()
    errs = np.sqrt(h.variances())
    ret = ax.bar(
        edges[:-1],
        vals,
        width=widths,
        yerr=errs if yerr else None,
        label=plot_object.name,
        align="edge",
    )
    return ax

q = pickle.load(open('output/QCD/QCD_Total.pkl', 'rb'))
histos = q['histograms']
h = histos['nJets'][ { 'pT_300': 1, 'nJets_456': sum, 'noLeptons': 1, 'dRJets_24': 1, 'bTag_312': sum, 'bTag_313': 1, 'dRbb_312': sum, 'dRbb_313': 1 } ]
hn = h["QCDInclusive2018",:]
fig,ax = drawAs1DHist(h,cat_axis="dataset", yerr=True)
addPrelim(ax)
addTitles(ax,hn)
plt.savefig('nJets-n-1.png')

