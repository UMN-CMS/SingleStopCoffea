import awkward as ak
import hist

from analyzer.core import analyzerModule
from analyzer.modules.axes import *


@analyzerModule("dataset_category", categories="axis_cat", depends_on=["objects"])
def datasetCategory(events, analyzer):
  analyzer.histogram_builder.addCategory(dataset_axis, analyzer.setname)
  return events, analyzer


njets_axis = hist.axis.IntCategory([4, 5, 6], name="number_jets", label="NJets")
@analyzerModule("njets_category", depends_on=['objects'], categories="axis_cat")
def nJetsCategory(events, analyzer):
  analyzer.histogram_builder.addCategory(njets_axis, ak.num(events.good_jets, axis=1))
  return events, analyzer


@analyzerModule("jetpT300", depends_on=['objects'], categories='axis_cat')
def jetpTCategory(events, analyzer):
  good_jets = events.good_jets
  filled_jets = ak.pad_none(good_jets, 4, axis=1)
  a = hist.axis.IntCategory([0, 1], name = 'jetpT300', label = 'jetpT300')
  analyzer.histogram_builder.addCategory(a, ak.fill_none(filled_jets[:, 0].pt >= 300, False))
  return events, analyzer


@analyzerModule("nJets456", depends_on=['objects'], categories='axis_cat')
def nJetReqCategory(events, analyzer):
  good_jets = events.good_jets
  a = hist.axis.IntCategory([0, 1], name = 'nJets456', label = 'nJets456')
  analyzer.histogram_builder.addCategory(a, (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6))
  return events, analyzer

@analyzerModule("leptonVeto", depends_on=['objects'], categories='axis_cat')
def leptonCategory(events, analyzer):
  good_muons = events.good_muons
  good_electrons = events.good_electrons
  a = hist.axis.IntCategory([0, 1], name = 'leptonVeto', label = 'leptonVeto')
  analyzer.histogram_builder.addCategory(a, (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0))
  return events, analyzer

@analyzerModule("dRJets24", depends_on=['objects', 'event_level'], categories='axis_cat')
def dRJetsCategory(events, analyzer):
  good_jets = events.good_jets
  filled_jets = ak.pad_none(good_jets, 4, axis=1)
  top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
  a = hist.axis.IntCategory([0, 1], name = 'dRJets24', label = 'dRJets24')
  analyzer.histogram_builder.addCategory(a, (top_two_dr < 4) & (top_two_dr > 2))
  return events, analyzer


@analyzerModule("312Bs", depends_on=['objects', 'event_level'], categories='axis_cat')
def bTag312(events, analyzer):
  mediumBs = events.med_bs
  tightBs = events.tight_bs
  a = hist.axis.IntCategory([0, 1], name = '312Bs', label = '312Bs')
  analyzer.histogram_builder.addCategory(a, (ak.num(mediumBs) >= 2) & (ak.num(tightBs) >= 1))
  return events, analyzer

@analyzerModule("313Bs", depends_on=['objects', 'event_level'], categories='axis_cat')
def bTag313(events, analyzer):
  tightBs = events.tight_bs
  a = hist.axis.IntCategory([0, 1], name = '313Bs', label = '313Bs')
  analyzer.histogram_builder.addCategory(a, (ak.num(tightBs) >= 3))
  return events, analyzer

@analyzerModule("dRbb_312", depends_on=['objects', 'event_level'], categories='axis_cat')
def dRbb312(events, analyzer):
  mediumBs = events.med_bs
  mediumBs = ak.pad_none(mediumBs, 2, axis = 1)
  dRbb12 = ak.fill_none(mediumBs[:, 0].delta_r(mediumBs[:, 1]), 0)
  a = hist.axis.IntCategory([0, 1], name = 'dRbb_312', label = 'dRbb_312')
  analyzer.histogram_builder.addCategory(a, (dRbb12 >= 1))
  return events, analyzer

@analyzerModule("dRbb_313", depends_on=['objects', 'event_level'], categories='axis_cat')
def dRbb313(events, analyzer):
  tightBs = events.tight_bs
  tightBs = ak.pad_none(tightBs, 2, axis = 1)
  dRbb12 = ak.fill_none(tightBs[:, 0].delta_r(tightBs[:, 1]), 0)
  a = hist.axis.IntCategory([0, 1], name = 'dRbb_313', label = 'dRbb_313')
  analyzer.histogram_builder.addCategory(a, (dRbb12 >= 1))
  return events, analyzer

@analyzerModule("triggers", categories='selection', depends_on=['objects'])
def triggers(events, analyzer):
  selection = analyzer.selection
  selection.add("hlt", (events.HLT.PFHT1050 | events.HLT.AK8PFJet400_TrimMass30).to_numpy())
  events = analyzer.applySelection(events)
  return events, analyzer

@analyzerModule("Mu50Trigger", categories='selection', depends_on=['objects'])
def Mu50Trigger(events, analyzer):
  selection = analyzer.selection
  good_electrons = events.good_electrons
  good_muons = events.good_muons
  selection.add("muonTrigger", (events.HLT.Mu50).to_numpy())
  selection.add("electronVeto", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) > 0)).to_numpy())
  events = analyzer.applySelection(events)
  return events, analyzer

@analyzerModule("Mu50SoftDropPlotTrigger", categories='selection', depends_on=['objects'])
def Mu50SoftDropPlotTrigger(events, analyzer):
  selection = analyzer.selection
  good_electrons = events.good_electrons
  good_muons = events.good_muons
  fatjets = events.FatJet
  fatjets = fatjets[(abs(events.FatJet.eta) < 2.4)]
  selection.add("muonTrigger", (events.HLT.Mu50).to_numpy())
  selection.add("electronVeto", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) > 0)).to_numpy())
  selection.add("highPTJet", (ak.num(fatjets) > 0).to_numpy())
  events = analyzer.applySelection(events)
  return events, analyzer

@analyzerModule("pTTrigger", depends_on=['objects'], categories='axis_cat')
def pTTrigger(events, analyzer):
  pT400 = events.HLT.AK8PFJet400_TrimMass30
  a = hist.axis.IntCategory([0, 1], name = 'pT400', label = 'pT400')
  analyzer.histogram_builder.addCategory(a, pT400)
  return events, analyzer

@analyzerModule('fatJetpTPlot', depends_on=['objects', 'event_level'], categories='main')
def pTPlot(events, analyzer):
  fatjets = events.FatJet
  fatjets = fatjets[(events.FatJet.pt > 175) & (abs(events.FatJet.eta) < 2.4) & (events.FatJet.msoftdrop > 50)]
  mask = ak.num(fatjets, axis=1) > 0
  fatjets_mask = fatjets[mask]
  analyzer.H('pT1', pt_axis, fatjets_mask[:, 0].pt, mask = mask, name = f'p_T of leading AK8 jet', description = f'p_T of leading AK8 jet')
  return events, analyzer

@analyzerModule("softDropMassPlot", depends_on=['objects', 'event_level'], categories='main')
def mSoftDropMassPlot(events, analyzer):
	fatjets = events.FatJet
	fatjets = fatjets[(events.FatJet.pt > 175) & (abs(events.FatJet.eta) < 2.4)]
	mask = ak.num(fatjets, axis=1) > 0
	fatjets_mask = fatjets[mask]
	analyzer.H("mSoftDrop",
		softdrop_axis,
		fatjets_mask[:, 0].msoftdrop,
		mask = mask,
		name = f'p_T of leading AK8 jet',
		description = f'p_T of leading AK8 jet',
	)
	return events, analyzer

@analyzerModule("softDroppT2DPlot", depends_on=['objects', 'event_level'], categories='main')
def softDrop2DPlot(events, analyzer):
	fatjets = events.FatJet
	fatjets = fatjets[(events.FatJet.pt > 175) & (abs(events.FatJet.eta) < 2.4)]
	mask = ak.num(fatjets, axis=1) > 0
	fatjets_mask = fatjets[mask]
	analyzer.H(rf"mSoftDrop2D",
		[pt_axis, softdrop_axis],
		[fatjets_mask[:, 0].pt, fatjets_mask[:, 0].msoftdrop],
		mask = mask,
		name = f'p_T of leading AK8 jet',
		description = f'p_T of leading AK8 jet',
	)
	return events, analyzer

@analyzerModule("HTTrigger", depends_on=['objects'], categories='cat_axis')
def HTTrigger(events, analyzer):
  HT1050 = events.HLT.PFHT1050
  a = hist.axis.IntCategory([0, 1], name = 'HT1050', label = 'HT1050')
  analyzer.histogram_builder.addCategory(a, HT1050)
  return events, analyzer

@analyzerModule("HTTriggerPlot", depends_on=['objects', 'event_level'], categories='main')
def HTTriggerPlot(events, analyzer):
	analyzer.H('HT',
		ht_axis,
		events.HT,
		name = 'HT',
		description = 'HT',
	)
	return events, analyzer

@analyzerModule("pTOrHTTrigger", depends_on=['objects'], categories='cat_axis')
def pTOrHTTrigger(events, analyzer):
  triggers = (events.HLT.PFHT1050) | (events.HLT.AK8PFJet400_TrimMass30)
  a = hist.axis.IntCategory([0, 1], name = 'triggers', label = 'triggers')
  analyzer.histogram_builder.addCategory(a, triggers)
  return events, analyzer

'''
@analyzerModule("nMinusOnePlots", ModuleType.MainHist)
def nMinusOnePlots(events, hmaker):
	gj = events.good_jets
	mediumBs = events.med_bs
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
'''
