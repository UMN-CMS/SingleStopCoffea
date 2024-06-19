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

def s_over_root_b(hists, hists_background, xvar, true_mass, sum_range_width):
    signal_hist = hists[xvar]
    background_hist = hists_background[xvar]
    signal_data = signal_hist.to_numpy()[0]
    background_data, edges = background_hist.to_numpy()
    bin_centers = (edges[:-1] + edges[1:])/2
    sqrt_b = np.sqrt(background_data)
    s_over_root_b_points = np.nan_to_num(signal_data/sqrt_b)

    return _L2_norm(s_over_root_b_points[abs(np.add(bin_centers, -true_mass)) < sum_range_width/2]), bin_centers, s_over_root_b_points