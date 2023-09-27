from analyzer.core import analyzerModule, ModuleType
from .axes import *
import awkward as ak
from analyzer.utilities import angleToNPiToPi


@analyzerModule("signal_hists", ModuleType.MainHist, require_tags=["signal"])
def createSignalHistograms(events, hmaker):
    ret = {}
    chi_match_axis = hist.axis.IntCategory(
        [0, 1, 2, 3], name="num_matched_chi", label=r"Number of reco chi jets correct"
    )
    stop_match_axis = hist.axis.IntCategory(
        [0, 1, 2, 3, 4],
        name="num_matched_stop",
        label=r"Number of reco jets matched to stop child",
    )
    num_top_3_are_chi = ak.num(
        events.matched_jet_idx[:, 1:][events.matched_jet_idx[:, 1:] < 3], axis=1
    )
    num_top_4_are_stop = ak.num(
        events.matched_jet_idx[events.matched_jet_idx < 4], axis=1
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
    ret[f"num_top_4_jets_matched_stop_children"] = hmaker(
        stop_match_axis, num_top_4_are_stop, name="num_top_4_are_stop"
    )

    idx_axis = hist.axis.IntCategory(range(0, 8), name="Jet Idx", label=r"$\chi^{\pm}$ b matched jet idx")
    e = events.matched_jet_idx[:, 1]
    m = ~ak.is_none(e)
    e=e[m]
    ret[f"chi_b_jet_idx"] = hmaker(idx_axis, e, name="chi_b_jet_idx", mask=m)

    idx_axis = hist.axis.IntCategory(range(0, 8), name="Jet Idx", label=r"$\tilde{t}$ b matched jet idx")
    e = events.matched_jet_idx[:, 0]
    m = ~ak.is_none(e)
    e=e[m]
    ret[f"stop_b_jet_idx"] = hmaker(idx_axis, e, name="stop_b_jet_idx", mask=m)

    ret[f"chi_b_dr"] = hmaker(
        makeAxis(20, 0, 5, "$\\Delta R$ between $\\chi$ and b from $\\tilde{t}$"),
        events.SignalParticles.chi.delta_r(events.SignalParticles.stop_b),
        name="chi_b_dr",
    )

    ret[f"chi_b_phi"] = hmaker(
        makeAxis(50, -4, 4, "$\\Delta \\phi$ between $\\chi$ and b from $\\tilde{t}$"),
        abs(angleToNPiToPi(events.SignalParticles.chi.phi - events.SignalParticles.stop_b.phi)),
        name="chi_b_phi",
    )

    ret[f"chi_b_eta"] = hmaker(
        makeAxis(50, -4, 4, "$\\Delta \\eta$ between $\\chi$ and b from $\\tilde{t}$"),
        events.SignalParticles.chi.eta - events.SignalParticles.stop_b.eta,
        name="chi_b_phi",
    )

    return ret
