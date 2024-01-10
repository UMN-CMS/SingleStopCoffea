from analyzer.core import analyzerModule, ModuleType
import awkward as ak

@analyzerModule("baseline_selection", ModuleType.Selection)
def createSelection(events, selection):
    good_jets = events.good_jets
    fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    med_b = events.med_bs
    tight_b = events.tight_bs
    tight_top = events.tight_tops
    #selection = PackedSelection()
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    filled_med = ak.pad_none(med_b, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    #selection.add("trigger", (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6))
    if "HLT" in events.fields:
        selection.add("hlt", (events.HLT.PFHT1050 | events.HLT.AK8PFJet360_TrimMass30).to_numpy())
    selection.add("highptjet", (ak.fill_none(filled_jets[:, 0].pt > 300, False)).to_numpy())
    selection.add("jets", ((ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)).to_numpy())
    selection.add("0Lep", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)).to_numpy())
    selection.add("2bjet", (ak.num(med_b) >= 2).to_numpy())
    selection.add("1tightbjet", (ak.num(tight_b) >= 1).to_numpy())
    #selection.add("jet_dr", ((top_two_dr < 4) & (top_two_dr > 2)).to_numpy())
    selection.add("b_dr", (med_dr > 1).to_numpy())
    return selection

@analyzerModule("bbpt_selection", ModuleType.Selection)
def createSelection(events, selection):
    med_b = events.med_bs
    filled_med = ak.pad_none(med_b, 2, axis=1)
    bbpt = ak.fill_none((filled_med[:, 0] + filled_med[:, 1]).pt, False)
    selection.add("bbpt", (bbpt>200).to_numpy())
    return selection

@analyzerModule("cr_selection", ModuleType.Selection)
def createCRSelection(events, selection):
    good_jets = events.good_jets
    fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    med_b = events.med_bs
    tight_b = events.tight_bs
    tight_top = events.tight_tops
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
    if "HLT" in events.fields:
        selection.add("hlt", (events.HLT.PFHT1050 | events.HLT.AK8PFJet360_TrimMass30).to_numpy())
    selection.add("highptjet", (ak.fill_none(filled_jets[:, 0].pt > 300, False)).to_numpy())
    selection.add("jets", ((ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)).to_numpy())
    selection.add("0Lep", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)).to_numpy())
    selection.add("0looseb", (ak.num(loose_b) == 0).to_numpy())
    return selection
