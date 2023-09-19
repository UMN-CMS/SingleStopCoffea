import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.font_manager as font_manager
from mpl_toolkits.axes_grid1 import make_axes_locatable
import numpy as np
import hist
import pickle
from analyzer.plotting.core_plots import *
import argparse
import mplhep as hep

parser = argparse.ArgumentParser()
parser.add_argument('--plot', type = str)
parser.add_argument('--coupling', type = str, default = '313')
parser.add_argument('--QCD', action = 'store_true')
parser.add_argument('--TT', action = 'store_true')
parser.add_argument('--signal', action = 'store_true')
args = parser.parse_args()

plot = args.plot
coupling = args.coupling
isQCD = args.QCD
isTT = args.TT
isSignal = args.signal

hep.style.use(['CMS', 'fira', 'firamath'])
plt.rcParams["figure.figsize"] = [13.00, 10.00]
plt.rcParams["figure.autolayout"] = True

### Setting up selection
selection = {}
xLabel = ''

if plot == 'nJets':
	selection = { 'dataset': sum, 'jetpT300': sum, 'nJets456': sum, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': sum, '313Bs': sum, 'dRbb_312': sum, 'dRbb_313': sum }
	xLabel = r'$N_{j}$'

elif plot == 'dR12':
	selection = { 'dataset': sum, 'jetpT300': 1, 'nJets456': 1, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': sum, '313Bs': 1, 'dRbb_312': sum, 'dRbb_313': sum }
	xLabel = r'$\Delta R_{1, 2}$'	

elif plot == 'pT1':
	selection = { 'dataset': sum, 'jetpT300': sum, 'nJets456': 1, 'leptonVeto': 1, 'dRJets24': 1, '312Bs': sum, '313Bs': 1, 'dRbb_312': sum, 'dRbb_313': 1}
	xLabel = r'$p_{T, 1}$'

<<<<<<< HEAD
elif plot == 'dRbb12':
	selection = { 'dataset': sum, 'jetpT300': 1, 'nJets456': 1, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': sum, '313Bs': 1, 'dRbb_312': sum, 'dRbb_313': sum}
	xLabel = r'$\Delta R_{b_{1}, b_{2}}$'

if coupling == '312':
	selection['312Bs'] = 1
	selection['313Bs'] = sum
	selection['dRbb_312'] = 1 if plot != 'dRbb12' else sum
=======
if coupling == '312':
	selection['312Bs'] = 1
	selection['313Bs'] = sum
	selection['dRbb_312'] = 1
>>>>>>> 013a0a610140cab61b55ac8aa1e10ec16d3665a0
	selection['dRbb_313'] = sum

if isQCD:
	### Adding QCD Inclusive
	q = pickle.load(open('output/QCD/QCD_Total.pkl', 'rb'))
	histos = q['histograms']
	h_background = histos[plot][selection]

	fig, ax = plt.subplots()
	ax.step(h_background.axes[0].edges[:-1], 
					np.divide(h_background.values(), sum(h_background.values())), 
					where = 'post', label = 'QCD', color = 'black', lw = 2)

if isTT:
	### Adding TTBar
	q = pickle.load(open('output/TT/TT2018.pkl', 'rb'))
	histos = q['histograms']
	h_background = histos[plot][selection]

	ax.step(h_background.axes[0].edges[:-1], 
					np.divide(h_background.values(), sum(h_background.values())), 
					where = 'post', label = r'$t \bar{t}$', color = 'black', linestyle = '--', lw = 2)

if isSignal:
	### Adding signal
	colors = ['red', 'green', 'blue']
	signals = ['1200_400', '1500_900', '2000_1900'] if coupling == '312' else ['1000_400', '1500_900', '2000_1900']
	for i, m in enumerate(signals):
		s = pickle.load(open('output/signal313/signal_{}.pkl'.format(m), 'rb')) if coupling == '313' else pickle.load(open('output/signal312/signal_{}.pkl'.format(m), 'rb'))
		histos = s['histograms']
		h_signal = histos[plot][selection]
		masses = m.split('_')
		ax.step(h_signal.axes[0].edges[:-1], 
						np.divide(h_signal.values(), sum(h_signal.values())), 
						where = 'post', 
						label = r'$m_{{\tilde{{t}}}} = {0} \ GeV, m_{{\tilde{{\chi}}^{{\pm}}_{1}}} = {2} \ GeV$'.format(masses[0], '1', masses[1]), 
						color = colors[i], lw = 2)

### Saving plot
if plot != 'nJets': ax.legend()
else: ax.legend(bbox_to_anchor = (1, 0.8))
ax.set_xlabel(xLabel)
ax.set_ylabel('Proportion of events')
plt.xlim(left = 0)
plt.ylim(bottom = 0)
if plot == 'nJets':
	plt.xticks(h_signal.axes[0].edges[:-1]),
	plt.minorticks_off()
	plt.tight_layout()
plt.savefig('plots/{0}/{1}-n-1.pdf'.format(coupling, plot))
