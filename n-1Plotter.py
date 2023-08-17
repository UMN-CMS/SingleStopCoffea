import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import hist
import pickle
from analyzer.plotting.core_plots import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--plot', type = str)
parser.add_argument('--coupling', type = str, default = '313')
args = parser.parse_args()

plot = args.plot
coupling = args.coupling

### Setting up selection
selection = {}

if plot == 'nJets':
	selection = { 'dataset': sum, 'jetpT300': sum, 'nJets456': sum, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': sum, '313Bs': sum, 'dRbb_312': sum, 'dRbb_313': sum }
elif plot == 'dR12':
	selection = { 'dataset': sum, 'jetpT300': 1, 'nJets456': 1, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': sum, '313Bs': 1, 'dRbb_312': sum, 'dRbb_313': 1 }
elif plot == 'pT1':
	selection = { 'dataset': sum, 'jetpT300': sum, 'nJets456': 1, 'leptonVeto': 1, 'dRJets24': 1, '312Bs': sum, '313Bs': 1, 'dRbb_312': sum, 'dRbb_313': 1}
if coupling == '312':
	selection['312Bs'] = 1
	selection['313Bs'] = sum
	selection['dRbb_312'] = 1
	selection['dRbb_313'] = sum

### Adding QCD Inclusive
q = pickle.load(open('output/QCD/QCD_Total.pkl', 'rb'))
histos = q['histograms']
h_background = histos[plot][selection]

fig, ax = plt.subplots()
ax.step(h_background.axes[0].edges[:-1], np.divide(h_background.values(), sum(h_background.values())), where = 'post', label = 'QCDInclusive2018', color = 'black')

### Adding TTBar
q = pickle.load(open('output/TT/TT2018.pkl', 'rb'))
histos = q['histograms']
h_background = histos[plot][selection]

fig, ax = plt.subplots()
ax.step(h_background.axes[0].edges[:-1], np.divide(h_background.values(), sum(h_background.values())), where = 'post', label = 'TT2018')

### Adding signal
colors = ['red', 'green', 'blue']
for i, m in enumerate(['1000_400', '1500_900', '2000_1900']):
	s = pickle.load(open('output/signal313/signal_{}.pkl'.format(m), 'rb')) if coupling == '313' else pickle.load(open('output/signal312/signal_{}.pkl'.format(m), 'rb'))
	histos = s['histograms']
	h_signal = histos[plot][selection]
	ax.step(h_signal.axes[0].edges[:-1], np.divide(h_signal.values(), sum(h_signal.values())), where = 'post', label = 'signal_{0}'.format(m), color = colors[i])

### Saving plot
ax.legend()
plt.savefig('{}-n-1.png'.format(plot))


