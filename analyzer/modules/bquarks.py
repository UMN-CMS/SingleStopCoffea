from analyzer.core import analyzerModule, ModuleType
from analyzer.math_funcs import angleToNPiToPi
from .axes import *
import awkward as ak
from .objects import b_tag_wps
import itertools as it
from .utils import numMatching

@analyzerModule("b_hists", ModuleType.MainHist)
def createBHistograms(events, hmaker):
    ret = {}
    l_bjets = events.loose_bs

    ret[f"loose_bjet_pt"] = hmaker(pt_axis, l_bjets.pt, name="Loose BJet $p_{T}$")
    ret[f"loose_nb"] = hmaker(b_axis, ak.num(l_bjets.pt), name="Loose BJet Count")
    for i in range(0, 4):
        mask = ak.num(l_bjets, axis=1) > i
        ret[f"loose_b_{i}_pt"] = hmaker(
            makeAxis(20, 0, 5, f"$p_{{T}}$ of rank  {i} loose b jet"),
            l_bjets[mask][:, i].pt,
            mask=mask,
            name=f"Loose BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(l_bjets, axis=1) > 1
    top2 = l_bjets[mask]
    lb_eta = abs(top2[:, 0].eta - top2[:, 1].eta)
    lb_phi = abs(angleToNPiToPi(top2[:, 0].phi - top2[:, 1].phi))
    lb_dr = top2[:, 0].delta_r(top2[:, 1])

    ret[f"loose_bb_eta"] = hmaker(
        makeAxis(20, 0, 5, "$\\Delta \\eta$ between leading loose b jets"),
        lb_eta,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_phi"] = hmaker(
        makeAxis(25, 0, 4, "$\\Delta \\phi$ between leading loose b jets"),
        lb_phi,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between the two highest rank loose b jets",
    )
    ret[f"loose_bdr"] = hmaker(
        makeAxis(20, 0, 5, "$\\Delta R$ between leading 2 loose b jets"),
        lb_dr,
        name=rf"Loose BJet $\Delta R$",
        description=rf"$\Delta R$ between the top 2 $p_T$ b jets",
    )

    m_bjets = events.med_bs
    ret[f"medium_bjet_pt"] = hmaker(pt_axis, m_bjets.pt, name="Medium BJet $p_{T}$")
    ret[f"medium_nb"] = hmaker(b_axis, ak.num(m_bjets.pt), name="Medium BJet Count")
    for i in range(0, 4):
        mask = ak.num(m_bjets, axis=1) > i
        ret[f"medium_b_{i}_pt"] = hmaker(
            makeAxis(20, 0, 5, f"$p_{{T}}$ of rank  {i} medium b jet"),
            m_bjets[mask][:, i].pt,
            mask=mask,
            name=f"Medium BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(m_bjets, axis=1) > 1
    top2 = m_bjets[mask]
    mb_eta = abs(top2[:, 0].eta - top2[:, 1].eta)
    mb_phi = abs(angleToNPiToPi(top2[:, 0].phi - top2[:, 1].phi))
    mb_dr = top2[:, 0].delta_r(top2[:, 1])

    ret[f"medium_bb_eta"] = hmaker(
        makeAxis(20, 0, 5, "$\\Delta \\eta$ between leading medium b jets"),
        mb_eta,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between the two highest rank medium b jets",
    )
    ret[f"medium_bb_phi"] = hmaker(
        makeAxis(25, 0, 4, "$\\Delta \\phi$ between leading medium b jets"),
        mb_phi,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between the two highest rank medium b jets",
    )
    ret[f"medium_bdr"] = hmaker(
        makeAxis(20, 0, 5, "$\\Delta R$ between leading 2 medium b jets"),
        mb_dr,
        name=rf"Medium BJet $\Delta R$",
        description=rf"$\Delta R$ between the top 2 $p_T$ b jets",
    )
    inv = top2[:, 0] + top2[:, 1]
    ret[f"medium_b_m"] = hmaker(
        makeAxis(60, 0, 3000, f"$m_{{bb}}", unit="GeV"),
        inv.mass,
        name=rf"medbmass",
    )
    ret[f"medium_b_pt"] = hmaker(
        makeAxis(20, 0, 1000 , f"$p_{{T, bb}}$", unit="GeV"),
        inv.pt,
        name=rf"medbmass",
    )

    return ret


@analyzerModule("b_ordinality_hists", ModuleType.MainHist)
def createBHistograms(events, hmaker):
    ret = {}
    gj = events.good_jets
    idx = ak.local_index(gj, axis=1)
    med_bjet_mask = gj.btagDeepFlavB > b_tag_wps[1]
    t_lead_b_idx = idx[med_bjet_mask]
    lead_b_idx = t_lead_b_idx[:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]

    ordax = hist.axis.Regular(10, 0, 10, name="Jet Rank", label=r"Jet Rank")

    ret[f"lead_medium_bjet_ordinality"] = hmaker(
        makeAxis(6, 1, 7, "Leading Medium B Jet Rank"),
        lead_b_idx + 1,
        name="Leading $p_{T}$ Medium B Jet Rank",
    )
    ret[f"sublead_medium_bjet_ordinality"] = hmaker(
        makeAxis(6, 1, 7, "SubLeading Medium B Jet Rank"),
        sublead_b_idx + 1,
        name="Subleading $p_{T}$ Medium B Jet Rank",
    )

    return ret
