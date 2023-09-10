import awkward as ak
from analyzer.core import analyzerModule, ModuleType
from analyzer.modules.axes import *
import hist



@analyzerModule("dataset_category", ModuleType.Categories)
def datasetCategory(events, data):
    return (dataset_axis, data["CatDataset"])

@analyzerModule("njets_category", ModuleType.Categories)
def njetCategory(events,data):
    a = hist.axis.IntCategory([4, 5, 6], name="number_jets", label="NJets")
    return (a, ak.num(events.good_jets, axis=1))

@analyzerModule("jetpT_category", ModuleType.Categories)
def jetpTCategory(events, data):
	good_jets = events.good_jets
	filled_jets = ak.pad_none(good_jets, 4, axis=1)
	a = hist.axis.IntCategory([0, 1], name = 'jetpT_300', label = 'jetpT300')
	return (a, ak.fill_none(filled_jets[:, 0].pt > 300, False))

@analyzerModule("nJetRequirement_category", ModuleType.Categories)
def nJetReqCategory(events, data):
	good_jets = events.good_jets
	a = hist.axis.IntCategory([0, 1], name = 'nJetRequirement', label = 'nJets')
	return (a, (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6))


@analyzerModule("lepton_category", ModuleType.Categories)
def leptonCategory(events, data):
	good_muons = events.good_muons
	good_electrons = events.good_electrons
	a = hist.axis.IntCategory([0, 1], name = 'is_lepton', label = 'isLepton')
	return (a, (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0))

@analyzerModule("dRJets_category", ModuleType.Categories)
def dRJetsCategory(events, data):
	good_jets = events.good_jets
	filled_jets = ak.pad_none(good_jets, 4, axis=1)
	top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
	a = hist.axis.IntCategory([0, 1], name = 'dRJets', label = 'dRJets')
	return (a, (top_two_dr < 4) & (top_two_dr > 2))

@analyzerModule("bSelection", ModuleType.Selection)
def bSelection(events, selection):
	mediumBs = events.med_bs
	tightBs = events.tight_bs
	selection.add("3bjet", ((ak.num(tightBs) >= 3).to_numpy()))
	print("tight bs")
	tightBs = ak.pad_none(tightBs, 2, axis = 1)
	dRbb12 = ak.fill_none(tightBs[:, 0].delta_r(tightBs[:, 1]), False)
	selection.add('dRbb12', (dRbb12 < 1).to_numpy())
	print("dRbb12")
	selection.add("hlt", (events.HLT.PFHT1050 | events.HLT.AK8PFJet400_TrimMass30).to_numpy())
	print("HT trigger")
	return selection
