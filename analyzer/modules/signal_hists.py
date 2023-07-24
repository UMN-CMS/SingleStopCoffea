from analyzer.core import analyzerModule, ModuleType
from .axes import *
import awkward as ak

@analyzerModule("signal_hists", ModuleType.MainHist, require_tags=["signal"])
def createSignalHistograms(events, hmaker):
    ret = {}
    chi_match_axis = hist.axis.IntCategory(
        [0, 1, 2, 3], name="num_matched_chi", label=r"Number of reco chi jets correct"
    )
    num_top_3_are_chi = ak.num(
        events.matched_jet_idx[:, 1:][events.matched_jet_idx[:, 1:] < 3], axis=1
    )
    num_sub_3_are_chi = ak.num(
        events.matched_jet_idx[:, 1:][
            (events.matched_jet_idx[:, 1:] < 4) & (events.matched_jet_idx[:, 1:] > 0)
        ],
        axis=1,
    )
    ret[f"num_top_3_jets_matched_chi_children"] = hmaker(
        chi_match_axis, num_top_3_are_chi, name="num_top_3_are_chi"
    )
    ret[f"num_sub_3_jets_matched_chi_children"] = hmaker(
        chi_match_axis, num_sub_3_are_chi, name="num_sub_3_are_chi"
    )

    idx_axis = hist.axis.IntCategory(range(0, 8), name="Jet Idx", label=r"Jet idx")
    e = events.matched_jet_idx[:, 1]
    ret[f"chi_b_jet_idx"] = hmaker(idx_axis, ak.fill_none(e, 7), name="chi_b_jet_idx")
    return ret
