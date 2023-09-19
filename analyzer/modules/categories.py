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

@analyzerModule("jetpT300", ModuleType.Categories)
def jetpTCategory(events, data):
	good_jets = events.good_jets
	filled_jets = ak.pad_none(good_jets, 4, axis=1)
	a = hist.axis.IntCategory([0, 1], name = 'jetpT300', label = 'jetpT300')
	return (a, ak.fill_none(filled_jets[:, 0].pt >= 300, False))

@analyzerModule("nJets456", ModuleType.Categories)
def nJetReqCategory(events, data):
	good_jets = events.good_jets
	a = hist.axis.IntCategory([0, 1], name = 'nJets456', label = 'nJets456')
	return (a, (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6))


@analyzerModule("leptonVeto", ModuleType.Categories)
def leptonCategory(events, data):
	good_muons = events.good_muons
	good_electrons = events.good_electrons
	a = hist.axis.IntCategory([0, 1], name = 'leptonVeto', label = 'leptonVeto')
	return (a, (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0))

@analyzerModule("dRJets24", ModuleType.Categories)
def dRJetsCategory(events, data):
	good_jets = events.good_jets
	filled_jets = ak.pad_none(good_jets, 4, axis=1)
	top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
	a = hist.axis.IntCategory([0, 1], name = 'dRJets24', label = 'dRJets24')
	return (a, (top_two_dr < 4) & (top_two_dr > 2))

@analyzerModule("312Bs", ModuleType.Categories)
def bTag312(events, data):
	mediumBs = events.med_bs
	tightBs = events.tight_bs
	a = hist.axis.IntCategory([0, 1], name = '312Bs', label = '312Bs')
	return (a, (ak.num(mediumBs) >= 2) & (ak.num(tightBs) >= 1))

@analyzerModule("313Bs", ModuleType.Categories)
def bTag313(events, data):
	tightBs = events.tight_bs
	a = hist.axis.IntCategory([0, 1], name = '313Bs', label = '313Bs')
	return (a, (ak.num(tightBs) >= 3))

@analyzerModule("dRbb_312", ModuleType.Categories)
def dRbb312(events, data):
	mediumBs = events.med_bs
	mediumBs = ak.pad_none(mediumBs, 2, axis = 1)
	dRbb12 = ak.fill_none(mediumBs[:, 0].delta_r(mediumBs[:, 1]), 0)
	a = hist.axis.IntCategory([0, 1], name = 'dRbb_312', label = 'dRbb_312')
	return (a, (dRbb12 >= 1))

@analyzerModule("dRbb_313", ModuleType.Categories)
def dRbb313(events, data):
	tightBs = events.tight_bs
	tightBs = ak.pad_none(tightBs, 2, axis = 1)
	dRbb12 = ak.fill_none(tightBs[:, 0].delta_r(tightBs[:, 1]), 0)
	a = hist.axis.IntCategory([0, 1], name = 'dRbb_313', label = 'dRbb_313')
	return (a, (dRbb12 >= 1))

@analyzerModule("triggers", ModuleType.Selection)
def triggers(events, selection):
	selection.add("hlt", (events.HLT.PFHT1050 | events.HLT.AK8PFJet400_TrimMass30).to_numpy())
	return selection

@analyzerModule("nMinusOnePlots", ModuleType.MainHist)
def nMinusOnePlots(events, hmaker):
	gj = events.good_jets
	ret = {}

	mask = ak.num(gj, axis=1) > 0
	gj_mask = gj[mask]
	ret[rf"pT1"] = hmaker(
		pt_axis,
		gj_mask[:, 0].pt,
		mask = mask,
		name = f'p_T of jet 1',
		description = f'p_T of jet 1',
	)

	mask = ak.num(gj, axis=1) > 1
	gj_mask = gj[mask]
	d_r = gj_mask[:, 0].delta_r(gj_mask[:, 1])
	ret[rf"dR12"] = hmaker(
		dr_axis,
		d_r,
		mask = mask,
		name = rf'$\Delta R$ between jets 1 and 2',
		description = rf'$\Delta R$ between jets 1 and 2',
	)

	ret[rf'nJets'] = hmaker(
		tencountaxis,
		ak.num(gj, axis=1),
		name = rf'Number of jets',
		description = rf'Number of jets',
	)

	ret[rf'HT'] = hmaker(
		ht_axis,
		events.HT,
		name = 'HT',
		description = 'HT',
	)


	b_mask = ak.num(mediumBs) > 2
	maskedBs = mediumBs[b_mask]
	dRbb12 = maskedBs[:, 0].delta_r(maskedBs[:, 1])
	ret[rf'dRbb12'] = hmaker(
		dr_axis,
		dRbb12,
		mask = b_mask,
		name = rf'$\Delta R$ between medium-b-tagged jets 1 and 2',
		description = rf'$\Delta R$ between medium-b-tagged jets 1 and 2',
	)

	return ret
