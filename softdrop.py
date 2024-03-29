import pickle
import sys
import matplotlib as mpl
import matplotlib.pyplot as plt
from analyzer.plotting.plots_1d import *
from analyzer.plotting.plots_2d import *
from analyzer.plotting.mplstyles import *
from analyzer.plotting.annotations import *
from analyzer.plotting.plottables import *
from analyzer.modules.axes import *
import numpy as np


sys.path.append(".")

filebase = 'signal_312_'
files = ['1200_400', '1200_900', '1200_1100', '1500_400', '1500_900', '1500_1400']
files = ['1500_400']
outputDir = 'SoftDropPlots/'
loadStyles()


for f in files:
	path = filebase + f + '.pkl'
	d = pickle.load(open(path, 'rb'))

	#mSoftDropHist = d.results[filebase + f].histograms['mSoftDrop'][{'dataset': filebase + f, 'HT1050': 0, 'pT400': sum}]
	#mSoftDrop2DHist = d.results[filebase + f].histograms['mSoftDrop2D'][{'dataset': filebase + f, 'HT1050': 0, 'pT400': sum}]

	mSoftDropHist = d.results[filebase + f].histograms['mSoftDrop'][{'dataset': filebase + f, 'triggers': 1}]
	mSoftDrop2DHist = d.results[filebase + f].histograms['mSoftDrop2D'][{'dataset': filebase + f, 'triggers': 1}]

	softDrop1D_PO = PlotObject.fromHist(mSoftDropHist, title = r'm_{Soft Drop}')
	softDrop2D_PO = PlotObject.fromHist(mSoftDrop2DHist, title = r'm_{Soft Drop}')

	fig, ax = plt.subplots()
	drawAs1DHist(ax, softDrop1D_PO)
	addPrelim(ax)
	addTitles1D(ax, softDrop1D_PO)
	plt.title(filebase + f)
	fig.savefig(outputDir + 'mSoftDrop_' + filebase + f + '.pdf')
	plt.close(fig)

	
	fig, ax = plt.subplots()
	drawAs2DHist(ax, softDrop2D_PO)
	addPrelim(ax)
	addTitles2D(ax, softDrop2D_PO)
	plt.title(filebase + f)
	fig.savefig(outputDir + 'mSoftDroppT2DpT2D_' + filebase + f + '.pdf')
	plt.close(fig)
