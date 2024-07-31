import pickle
import analyzer
from analyzer.datasets import SampleManager
from analyzer.core import AnalysisResult
import numpy as np
from scipy.optimize import curve_fit
# import matplotlib.pyplot as plt
# from hist import Hist, new, axis
from analyzer.plotting import PlotObject, drawAs1DHist

def _L2_norm(values):
    sum_of_squares = np.sum(np.power(values, 2))
    return np.sqrt(sum_of_squares)

def gaussian(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

    
def s_over_root_b(hists, hists_background, xvar, true_mass, window_width=-1):
    signal_hist = hists[xvar]
    background_hist = hists_background[xvar]["Skim_QCDInclusive2018"]
    signal_data = signal_hist.to_numpy()[0]
    background_data, edges = background_hist.to_numpy()
    bin_centers = (edges[:-1] + edges[1:])/2
    sqrt_b = np.sqrt(background_data)
    s_over_root_b_points = np.nan_to_num(signal_data/sqrt_b)

    if window_width == -1: # no window
        return _L2_norm(s_over_root_b_points), bin_centers, s_over_root_b_points
    return _L2_norm(s_over_root_b_points[abs(np.add(bin_centers, -true_mass)) < window_width/2]), bin_centers, s_over_root_b_points

def significance_2D(hists, hists_background, xvar):
    signal_hist = hists[xvar]
    # print(hists_background)
    # print("-")
    # print(hists_background[xvar]["Skim_QCDInclusive2018"])
    background_hist = hists_background[xvar]["Skim_QCDInclusive2018"]
    signal_data = signal_hist.to_numpy()[0]
    # print(signal_data)
    background_data, edges_x, edges_y = background_hist.to_numpy()

    sum_term = np.add(signal_data, background_data) # S+B
    quotient_term = np.nan_to_num(np.divide(signal_data, background_data), posinf=0, neginf=0)
    log_term = np.log(1 + quotient_term) # ln(1 + S/B)
    sqrt_arg = np.clip(2 * np.add(np.multiply(sum_term, log_term), -signal_data), 0, np.inf)
    significance_points = np.sqrt(sqrt_arg)

    return _L2_norm(significance_points), edges_x, edges_y, significance_points
    # return _L2_norm(s_over_root_b_points[abs(np.add(bin_centers_x, -true_mass)) < window_width/2]), bin_centers, s_over_root_b_points