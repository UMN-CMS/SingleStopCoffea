from coffea.analysis_tools import PackedSelection
import awkward as ak

def createSelection(events):
    good_jets = events.good_jets
    fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    tight_top = events.tight_tops
    selection = PackedSelection()
    filled_jets = ak.pad_none(good_jets, 2, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
    selection.add("jets", (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 5))
    selection.add("0Lep", (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0))
    selection.add("2bjet", ak.num(loose_b) >= 2)
    selection.add("highptjet", ak.fill_none(filled_jets[:, 0].pt > 300, False))
    selection.add("jet_dr", (top_two_dr < 4) & (top_two_dr > 2))
    return selection
