import pickle
import analyzer
from analyzer.datasets import SampleManager
from analyzer.core import AnalysisResult
import numpy as np
# import matplotlib.pyplot as plt
# from hist import Hist, new, axis
from analyzer.plotting import PlotObject, drawAs1DHist

def _L2_norm(values):
    sum_of_squares = np.sum(np.power(values, 2))
    return np.sqrt(sum_of_squares)

def s_over_root_b(hists, background_name, signal_name, xvar, true_mass):#, plot=False):
    signal_hist = hists[xvar]
    background_hist = hists[xvar]
    signal_data = signal_hist.to_numpy()[0]
    background_data, edges = background_hist.to_numpy()
    bin_centers = (edges[:-1] + edges[1:])/2
    sqrt_b = np.sqrt(background_data)
    result = np.nan_to_num(signal_data/sqrt_b)

    # if plot:
    #     background_style = s.getCollection(background_name).style
    #     signal_style = s.getSet(signal_name).style
    #     background_obj = PlotObject.fromHist(hists[xvar][background_name,...], style=background_style)
    #     signal_obj = PlotObject.fromHist(hists[xvar][signal_name,...], style=signal_style)
    #     fig, ax = plt.subplots(1, 2)
    #     fig.set_figwidth(12)
    #     ax[0].set_xlabel(xvar)
    #     ax[0].set_ylabel("number of events")
    #     ax[0].set_yscale("log")
    #     ax[1].set_xlabel(xvar)
    #     ax[1].set_ylabel("$S/\sqrt{B}$")
    #     # ax.set_xlabel(key)
    #     # ax.set_ylabel("$S/\sqrt{B}$")
    #     # ax.set_title("Scatter plot of S/sqrt(B)")
    #     drawAs1DHist(ax[0], background_obj)
    #     drawAs1DHist(ax[0], signal_obj)
    #     ax[1].scatter(bin_centers, result)

    return _L2_norm(result[abs(np.add(bin_centers, -true_mass)) < 150])