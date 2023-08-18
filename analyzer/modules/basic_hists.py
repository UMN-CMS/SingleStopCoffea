from analyzer.core import analyzerModule, ModuleType
from .axes import *
import awkward as ak
from .objects import b_tag_wps
import itertools as it

@analyzerModule("pre_sel_hists", ModuleType.PreSelectionHist)
def makePreSelectionHistograms(events, hmaker):
    if "LHE" not in events.fields:
        return {}
    ret = {}
    dataset = events.metadata["dataset"]
    w = events.EventWeight
    # ret[f"LHEHT"] = hmaker(
    #    ht_axis,
    #    events.LHE.HT,
    #    w,
    #    name="Event LHE HT Preselection",
    #    description="HT of the LHE level event before any selections are applied",
    # )
    return ret




@analyzerModule("event_level_hists", ModuleType.MainHist)
def createEventLevelHistograms(events, hmaker):
    dataset = events.metadata["dataset"]
    ret = {}
    ret[f"HT"] = hmaker(
        ht_axis,
        events.HT,
        name="Event HT",
        description="Sum of $p_T$ of good AK4 jets.",
    )
    if "LHE" not in events.fields:
        return ret
    ret[f"nQLHE"] = hmaker(
        tencountaxis,
        events.LHE.Nuds + events.LHE.Nc + events.LHE.Nb,
        name="Quark Count LHE",
        description="Number of LHE level Quarks",
    )
    ret[f"nJLHE"] = hmaker(
        tencountaxis,
        events.LHE.Njets,
        name="Jet Count LHE",
        description="Number of LHE level Jets",
    )
    ret[f"nGLHE"] = hmaker(
        tencountaxis,
        events.LHE.Nglu,
        name="Gluon Count LHE",
        description="Number of LHE level gluons",
    )
    return ret

@analyzerModule("chargino_hists", ModuleType.MainHist)
def charginoRecoHistograms(events, hmaker):
    ret = {}
    gj = events.good_jets
    w = events.EventWeight
    idx = ak.local_index(gj, axis=1)
    med_bjet_mask = gj.btagDeepFlavB > b_tag_wps[1]

    t_lead_b_idx = idx[med_bjet_mask]
    lead_b_idx = t_lead_b_idx[:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]
    no_lead_idxs = idx[idx != lead_b_idx]
    no_sublead_idxs = idx[idx != sublead_b_idx]
    no_lead_or_sublead_idxs = idx[(idx != sublead_b_idx) & (idx != lead_b_idx)]
    no_lead_jets = gj[no_lead_idxs]
    no_sublead_jets = gj[no_sublead_idxs]
    first, second = ak.unzip(ak.combinations(no_lead_jets[:, 0:3], 2))
    max_dr_no_lead = ak.max(first.delta_r(second), axis=1)
    first, second = ak.unzip(ak.combinations(no_sublead_jets[:, 0:3], 2))
    max_dr_no_sublead = ak.max(first.delta_r(second), axis=1)
    max_no_lead_over_max_sublead = max_dr_no_lead / max_dr_no_sublead

    jets = gj[:, 0:4].sum()
    m14= jets.mass

    uncomp_charg =  (no_lead_jets[:, 0:3].sum()).mass

    m14_axis = hist.axis.Regular(60, 0, 3000, name="m14", label=r"$m_{14}$ [GeV]")
    mchi_axis = hist.axis.Regular(60, 0, 3000, name="mchi", label=r"$m_{\chi}$ [GeV]")

    ret[f"m3_top_3_no_lead_b"] = hmaker(
        mchi_axis,
        uncomp_charg,
        name="Mass of Jets 1-3 Without Leading B",
    )
    ret[f"m14_vs_m3_top_3_no_lead_b"] = hmaker(
        [m14_axis,mchi_axis],
        [m14, uncomp_charg],
        name="$m_{14}$ vs Mass of Jets 1-3 Without Leading B",
    )

    ret[f"m3_top_3_no_b_unless_dR_charg_gt_2"] = hmaker(
        mchi_axis,
        ak.where(
            max_no_lead_over_max_sublead > 2,
            (no_lead_jets[:, 0:3].sum()).mass,
            (
                gj[no_lead_or_sublead_idxs][:, 0:2].sum()
                + gj[ak.singletons(lead_b_idx)][:, 0]
            ).mass,
        ),
        name="m3_top3_no_b_unless_dR_charg_gt_2",
    )

    comp_charg= (no_lead_jets[:, 0:2].sum() + gj[ak.singletons(lead_b_idx)][:, 0]).mass

    ret[f"m3_top_2_plus_lead_b"] = hmaker(
        mchi_axis,
        uncomp_charg,
        name="m3_top_2_plus_lead_b",
    )

    ret[f"m14_vs_m3_top_2_plus_lead_b"] = hmaker(
        [m14_axis,mchi_axis],
        [m14, comp_charg],
        name="Mass of Top 2 $p_T$ Jets Plus Leading b Jet",
    )
    ratio_axis = hist.axis.Regular(
            50,
            0,
            1,
            name=f"ratio",
            label=rf"$\frac{{m_{{ \chi }} }}{{ m_{{ 14 }} }}$ [GeV]",
        )

    ret[f"ratio_m14_vs_m3_top_2_plus_lead_b"] = hmaker(
        [m14_axis, ratio_axis],
        [m14, comp_charg/m14],
        name="ratio_m14_vs_m3_top_2_plus_lead_b",
    )

    ret[f"ratio_m14_vs_m3_top_3_no_lead_b"] = hmaker(
        [m14_axis,ratio_axis],
        [m14, uncomp_charg / m14],
        name="ratio_m3_top_3_no_lead_b",
    )

    return ret


@analyzerModule("jet_hists", ModuleType.MainHist)
def createJetHistograms(events, hmaker):
    ret = {}
    dataset = events.metadata["dataset"]
    gj = events.good_jets
    w = events.EventWeight

    ret[f"h_njet"] = hmaker(nj_axis, ak.num(gj), name="njets")
    jet_combos = [(0, 4), (0, 3), (1, 4)]
    co = lambda x: it.combinations(x, 2)

    masses = {}

    for i, j in jet_combos:
        jets = gj[:, i:j].sum()
        masses[(i, j)] = jets.mass
        ret[f"m{i+1}{j}_pt"] = hmaker(
            pt_axis,
            jets.pt,
            name=f"Composite Jet {i+1} to Jet {j} $p_T$",
            description=f"$p_T$ of the sum of jets {i+1} to {j}",
        )
        ret[f"m{i+1}{j}_eta"] = hmaker(
            eta_axis,
            jets.eta,
            name=rf"Composite Jet {i+1} to Jet {j} $\eta$",
            description=rf"$\eta$ of the sum of jets {i+1} to {j}",
        )
        ret[rf"m{i+1}{j}_m"] = hmaker(
            mass_axis,
            jets.mass,
            name=rf"Composite Jet {i+1} to Jet {j} mass",
            description=rf"Mass of the sum of jets {i+1} to {j}",
        )

    for p1, p2 in co(jet_combos):
        p1_1, p1_2 = p1
        p2_1, p2_2 = p2
        m1 = hist.axis.Regular(
            60,
            0,
            3000,
            name=f"mass_{p1_1+1}{p1_2}",
            label=rf"$m_{{{p1_1 + 1}{p1_2}}}$ [GeV]",
        )
        m2 = hist.axis.Regular(
            60,
            0,
            3000,
            name=f"mass_{p2_1+1}{p2_2}",
            label=rf"$m_{{{p2_1 + 1}{p2_2}}}$ [GeV]",
        )
        ratio = hist.axis.Regular(
            50,
            0,
            1,
            name=f"Mass Ratio",
            label=rf"$\frac{{m_{{ \sum {p2_1+1}{p2_2} }} }}{{ m_{{ \sum {p1_1+1}{p1_2} }} }}$ [GeV]",
        )
        ret[f"m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}"] = hmaker(
            [m1, m2], [masses[p1], masses[p2]], name="Comp mass"
        )

        ret[f"ratio_m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}"] = hmaker(
            [m1, ratio], [masses[p1], masses[p2] / masses[p1]],
            name=f"ratio_m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}"
        )

    for i, j in jet_combos:
        m1 = hist.axis.Regular(
            60,
            0,
            3000,
            name=f"mass_{i}{j}",
            label=rf"$m_{{{i}{j}}}$ [GeV]",
        )
        ret[f"m{i}{j}_vs_pt4"] = hmaker(
            [m1, pt_axis], [masses[(i, j)], gj[:, 3].pt], name=f"m{i}{j}_vs_pt4"
        )
        ret[f"m{i}{j}_vs_pt1"] = hmaker(
            [m1, pt_axis], [masses[(i, j)], gj[:, 0].pt], name=f"m{i}{j}_vs_pt1"
        )

        ret[f"m{i}{j}_vs_HT"] = hmaker(
            [m1, pt_axis], [masses[(i, j)], events.HT], name=f"m{i}{j}_vs_HT"
        )

        ret[f"m{i}{j}_vs_lead_b"] = hmaker(
            [m1, pt_axis],
            [masses[(i, j)], events.med_bs[:, 0].pt],
            name=f"m{i}{j}_vs_lead_b_pt",
        )

        ret[f"m{i}{j}_vs_sublead_b"] = hmaker(
            [m1, pt_axis],
            [masses[(i, j)], events.med_bs[:, 1].pt],
            name=f"m{i}{j}_vs_sublead_b_pt",
        )

    for i in range(0, 4):
        ret[rf"pt_{i}"] = hmaker(
            pt_axis,
            gj[:, i].pt,
            name=f"$p_T$ of jet {i+1}",
            description=f"$p_T$ of jet {i+1} ",
        )
        ret[f"eta_{i}"] = hmaker(
            eta_axis,
            gj[:, i].eta,
            name=f"$\eta$ of jet {i+1}",
            description=f"$\eta$ of jet {i+1}",
        )
        ret[f"phi_{i}"] = hmaker(
            phi_axis,
            gj[:, i].phi,
            name=rf"$\phi$ of jet {i+1}",
            description=rf"$\phi$ of jet {i+1}",
        )

    padded_jets = ak.pad_none(gj, 5, axis=1)
    masks = {}
    for i, j in list(x for x in it.combinations(range(0, 5), 2) if x[0] != x[1]):
        mask = ak.num(gj, axis=1) > max(i, j)
        w_mask = w[mask]
        masked_jets = gj[mask]
        d_eta = masked_jets[:, i].eta - masked_jets[:, j].eta
        d_r = masked_jets[:, i].delta_r(masked_jets[:, j])
        d_phi = masked_jets[:, i].phi - masked_jets[:, j].phi
        masks[(i, j)] = mask
        ret[rf"d_eta_{i+1}_{j}"] = hmaker(
            eta_axis,
            d_eta,
            mask=mask,
            name=rf"$\Delta \eta$ between jets {i+1} and {j}",
            description=rf"$\Delta \eta$ between jets {i+1} and {j}",
        )
        ret[f"d_phi_{i+1}_{j}"] = hmaker(
            phi_axis,
            d_phi,
            mask=mask,
            name=rf"$\Delta \phi$ between jets {i+1} and {j}",
            description=rf"$\Delta \phi$ between jets {i+1} and {j}",
        )
        ret[f"d_r_{i+1}_{j}"] = hmaker(
            dr_axis,
            d_r,
            mask=mask,
            name=rf"$\Delta R$ between jets {i+1} and {j}",
            description=rf"$\Delta R$ between jets {i+1} and {j}",
        )

    for i in range(0, 5):
        mask = ak.num(gj, axis=1) > i
        masked_jets = gj[mask]
        htratio = masked_jets[:, i].pt / events.HT[mask]
        ret[f"pt_ht_ratio_{i+1}"] = hmaker(
            hist.axis.Regular(50, 0, 5, name="pt_o_ht", label=r"$\frac{p_{T}}{HT}$"),
            htratio,
            mask=mask,
            name=rf"Ratio of jet {i} $p_T$ to event HT",
            description=rf"Ratio of jet {i} $p_T$ to event HT",
        )
    for p1, p2 in co(co(range(0, 4))):
        mask = masks[p1] & masks[p2]
        p1_vals = gj[mask][:, p1[0]].phi - gj[mask][:, p1[1]].phi
        p2_vals = gj[mask][:, p2[0]].phi - gj[mask][:, p2[1]].phi
        ret["d_phi_{}{}_vs_{}{}".format(*p1, *p2)] = hmaker(
            [
                hist.axis.Regular(
                    50, 0, 5, name="dp1", label=r"$\Delta \phi_{" + f"{p1}" + r"}$"
                ),
                hist.axis.Regular(
                    50, 0, 5, name="dp2", label=r"$\Delta \phi_{" + f"{p2}" + r"}$"
                ),
            ],
            [p1_vals, p2_vals],
            mask=mask,
            name=rf"$\Delta \phi_{p1}$ vs $\Delta \phi_{p2}$",
        )
    return ret


@analyzerModule("tag_hists", ModuleType.MainHist)
def createTagHistograms(events, hmaker):
    ret = {}
    dataset = events.metadata["dataset"]
    gj = events.good_jets
    w = events.EventWeight
    for name, wp in it.product(("tops", "bs", "Ws"), ("loose", "med", "tight")):
        ret[f"{name}_{wp}"] = hmaker(
            tencountaxis,
            ak.num(events[f"{wp}_{name}"], axis=1),
            name=f"Number of {wp} {name}",
        )
    for name, wp in it.product(("deep_top", "deep_W"), range(1, 5)):
        ret[f"{name}_{wp}"] = hmaker(
            tencountaxis,
            ak.num(events[f"{name}_wp{wp}"], axis=1),
            name=f"Number of wp{wp} {name}",
        )

@analyzerModule("b_hists", ModuleType.MainHist)
def createBHistograms(events, hmaker):
    ret = {}
    dataset = events.metadata["dataset"]
    l_bjets = events.loose_bs

    ret[f"loose_bjet_pt"] = hmaker(pt_axis, l_bjets.pt, name="Loose BJet $p_{T}$")
    ret[f"loose_nb"] = hmaker(b_axis, ak.num(l_bjets.pt), name="Loose BJet Count")
    ret[f"loose_bdr"] = hmaker(
        b_axis,
        l_bjets[:, 0].delta_r(l_bjets[:, 1]),
        name=rf"Loose BJet $\Delta R$",
        description=rf"$\Delta R$ between the top 2 $p_T$ b jets",
    )
    for i in range(0, 4):
        mask = ak.num(l_bjets, axis=1) > i
        ret[f"loose_b_{i}_pt"] = hmaker(
            pt_axis,
            l_bjets[mask][:, i].pt,
            mask=mask,
            name=f"Loose BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(l_bjets, axis=1) > 1
    top2 = l_bjets[mask]
    lb_eta = top2[:, 0].eta - top2[:, 1].eta
    lb_phi = top2[:, 0].phi - top2[:, 1].phi
    lb_dr = top2[:, 0].delta_r(top2[:, 1])


    ret[f"loose_bb_eta"] = hmaker(
        eta_axis,
        lb_eta,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_phi"] = hmaker(
        phi_axis,
        lb_phi,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_deltar"] = hmaker(
        dr_axis,
        lb_dr,
        name=f"$\Delta R$ BB$",
        description=f"$\Delta R$ between the two highest rank loose b jets",
    )

    m_bjets = events.med_bs
    mask = ak.num(m_bjets, axis=1) > 1
    top2 = m_bjets[mask]
    mb_eta = top2[:, 0].eta - top2[:, 1].eta
    mb_phi = top2[:, 0].phi - top2[:, 1].phi
    mb_dr = top2[:, 0].delta_r(top2[:, 1])

    ret[f"medium_bjet_pt"] = hmaker(pt_axis, m_bjets.pt, name="Medium BJet $p_{T}$")
    ret[f"medium_nb"] = hmaker(b_axis, ak.num(m_bjets.pt), name="Medium BJet Count")
    ret[f"medium_bdr"] = hmaker(
        b_axis,
        m_bjets[:, 0].delta_r(m_bjets[:, 1]),
        name=rf"Medium BJet $\Delta R$",
        description=rf"$\Delta R$ between the top 2 $p_T$ b jets",
    )
    for i in range(0, 4):
        mask = ak.num(m_bjets, axis=1) > i
        ret[f"medium_b_{i}_pt"] = hmaker(
            pt_axis,
            m_bjets[mask][:, i].pt,
            mask=mask,
            name=f"Medium BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    ret[f"medium_bb_eta"] = hmaker(
        eta_axis,
        mb_eta,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between the two highest rank medium b jets",
    )
    ret[f"medium_bb_phi"] = hmaker(
        phi_axis,
        mb_phi,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between the two highest rank medium b jets",
    )
    ret[f"medium_bb_deltar"] = hmaker(
        dr_axis,
        mb_dr,
        name=f"$\Delta R$ BB$",
        description=f"$\Delta R$ between the two highest rank medium b jets",
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


    ret[f"lead_medium_bjet_ordinality"] = hmaker(ordax, lead_b_idx+1, name="Leading $p_{T}$ Medium B Jet Rank")
    ret[f"sublead_medium_bjet_ordinality"] = hmaker(ordax, sublead_b_idx+1, name="Subleading $p_{T}$ Medium B Jet Rank")
    
    return ret

