from analyzer.core import analyzerModule, ModuleType
from analyzer.math_funcs import angleToNPiToPi
from .axes import *
import awkward as ak
from .objects import b_tag_wps
import itertools as it


@analyzerModule("pre_sel_hists", ModuleType.PreSelectionHist)
def makePreSelectionHistograms(events, hmaker):
    if "LHE" not in events.fields:
        return {}
    ret = {}
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
    ret = {}
    ret[f"HT"] = hmaker(
        makeAxis(60, 0, 3000, "HT", unit="GeV"),
        events.HT,
        name="Event HT",
        description="Sum of $p_T$ of good AK4 jets.",
    )
    if "LHE" not in events.fields:
        return ret
    ret[f"nQLHE"] = hmaker(
        makeAxis(10, 0, 10, "Quark Count LHE"),
        events.LHE.Nuds + events.LHE.Nc + events.LHE.Nb,
        name="Quark Count LHE",
        description="Number of LHE level Quarks",
    )
    ret[f"nJLHE"] = hmaker(
        makeAxis(10, 0, 10, "Jet Count LHE"),
        events.LHE.Njets,
        name="Jet Count LHE",
        description="Number of LHE level Jets",
    )
    ret[f"nGLHE"] = hmaker(
        makeAxis(10, 0, 10, "Gluon Count LHE"),
        events.LHE.Nglu,
        name="Gluon Count LHE",
        description="Number of LHE level gluons",
    )
    return ret


@analyzerModule("perfect_hists", ModuleType.MainHist, require_tags=["signal"])
def genMatchingMassReco(events, hmaker):
    ret = {}
    mask = ~ak.any(ak.is_none(events.matched_jets, axis=1), axis=1)
    all_matched = events.matched_jets[mask]

    ret[f"mchi_gen_matched"] = hmaker(
        makeAxis(
            60,
            0,
            3000,
            r"Mass of Gen Matched Jets From $\tilde{\chi}^{\pm}$",
            unit="GeV",
        ),
        all_matched[:, 1:4].sum().mass,
        mask=mask,
        name="genmatchedchi",
    )
    ret[f"mstop_gen_matched"] = hmaker(
        makeAxis(
            60,
            0,
            3000,
            r"Mass of Gen Matched Jets From $\tilde{t}$",
            unit="GeV",
        ),
        all_matched[:, 0:4].sum().mass,
        mask=mask,
        name="Genmatchedm14",
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
    m14 = jets.mass

    uncomp_charg = (no_lead_jets[:, 0:3].sum()).mass

    m14_axis = makeAxis(60, 0, 3000, r"$m_{14}$ [GeV]")
    mchi_axis = makeAxis(60, 0, 3000, r"$m_{\chi}$ [GeV]")

    ret[f"m3_top_3_no_lead_b"] = hmaker(
        makeAxis(60, 0, 3000, r"Mass of Jets 1-3 without Leading b", unit="GeV"),
        uncomp_charg,
        name="Mass of Jets 1-3 Without Leading B",
    )
    ret[f"m14_vs_m3_top_3_no_lead_b"] = hmaker(
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, r"$m_{3 (no b)}$", unit="GeV"),
        ],
        [m14, uncomp_charg],
        name="$m_{14}$ vs Mass of Jets 1-3 Without Leading B",
    )

    # ret[f"m3_top_3_no_b_unless_dR_charg_gt_2"] = hmaker(
    #    mchi_axis,
    #    ak.where(
    #        max_no_lead_over_max_sublead > 2,
    #        (no_lead_jets[:, 0:3].sum()).mass,
    #        (
    #            gj[no_lead_or_sublead_idxs][:, 0:2].sum()
    #            + gj[ak.singletons(lead_b_idx)][:, 0]
    #        ).mass,
    #    ),
    #    name="m3_top3_no_b_unless_dR_charg_gt_2",
    # )

    comp_charg = (no_lead_jets[:, 0:2].sum() + gj[ak.singletons(lead_b_idx)][:, 0]).mass

    ret[f"m3_top_2_plus_lead_b"] = hmaker(
        makeAxis(
            60, 0, 3000, r"Mass of leading 2 $p_{T}$ Jets + leading b Jet", unit="GeV"
        ),
        comp_charg,
        name="m3_top_2_plus_lead_b",
    )

    ret[f"m14_vs_m3_top_2_plus_lead_b"] = hmaker(
        [
            makeAxis(60, 0, 3000, r"$m_{4}$", unit="GeV"),
            makeAxis(
                60,
                0,
                3000,
                r"Mass of leading 2 $p_{T}$ Jets + leading b Jet",
                unit="GeV",
            ),
        ],
        [m14, comp_charg],
        name="Mass of Top 2 $p_T$ Jets Plus Leading b Jet",
    )
    ratio_axis = hist.axis.Regular(
        50,
        0,
        1,
        name=f"ratio",
        label=r"$\frac{m_{\chi}}{m_{4}}$ [GeV]",
    )

    ret[f"ratio_m14_vs_m3_top_2_plus_lead_b"] = hmaker(
        [
            makeAxis(60, 0, 3000, r"$m_{4}$ [GeV]"),
            makeAxis(
                50,
                0,
                1,
                r"$\frac{m_{3 (incl b)}}{m_{4}}$",
            ),
        ],
        [m14, comp_charg / m14],
        name="ratio_m14_vs_m3_top_2_plus_lead_b",
    )

    ret[f"ratio_m14_vs_m3_top_3_no_lead_b"] = hmaker(
        [
            makeAxis(60, 0, 3000, r"$m_{4}$ [GeV]"),
            makeAxis(
                50,
                0,
                1,
                r"$\frac{m_{3 \mathrm{(no b)}}}{m_{4}}$",
            ),
        ],
        [m14, uncomp_charg / m14],
        name="ratio_m3_top_3_no_lead_b",
    )

    return ret


@analyzerModule("jet_hists", ModuleType.MainHist)
def createJetHistograms(events, hmaker):
    ret = {}
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
            makeAxis(
                100,
                0,
                1500,
                f"$p_T ( \\sum_{{n={i+1}}}^{{{j}}} jet_{{n}})$ ",
                unit="GeV",
            ),
            jets.pt,
            name=f"Composite Jet {i+1} to Jet {j} $p_T$",
            description=f"$p_T$ of the sum of jets {i+1} to {j}",
        )
        ret[f"m{i+1}{j}_eta"] = hmaker(
            makeAxis(20, -5, 5, f"$\eta ( \\sum_{{n={i+1}}}^{{{j}}} ) jet_{{n}}$"),
            jets.eta,
            name=rf"Composite Jet {i+1} to Jet {j} $\eta$",
            description=rf"$\eta$ of the sum of jets {i+1} to {j}",
        )
        mtitle = 4 if j - i == 4 else 3
        ret[rf"m{i+1}{j}_m"] = hmaker(
            makeAxis(60, 0, 3000, f"$m_{{{mtitle}}}$", unit="GeV"),
            jets.mass,
            name=rf"Composite Jet {i+1} to Jet {j} mass",
            description=rf"Mass of the sum of jets {i+1} to {j}",
        )

    for p1, p2 in co(jet_combos):
        p1_1, p1_2 = p1
        p2_1, p2_2 = p2
        mtitle1 = 4 if p1_2 - p1_1 == 4 else 3
        mtitle2 = 4 if p2_2 - p2_1 == 4 else 3
        ret[f"m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}"] = hmaker(
            [
                makeAxis(
                    60, 0, 3000, rf"$m_{{{mtitle1}}}$", unit="GeV", append_name="1"
                ),
                makeAxis(
                    60, 0, 3000, rf"$m_{{{mtitle2}}}$", unit="GeV", append_name="2"
                ),
            ],
            [masses[p1], masses[p2]],
            name="Comp mass",
        )

        ret[f"ratio_m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}"] = hmaker(
            [
                makeAxis(60, 0, 3000, rf"$m_{{{mtitle1}}}$", unit="GeV"),
                makeAxis(
                    50,
                    0,
                    1,
                    rf"$\frac{{m_{{ {mtitle2} }} }}{{ m_{{ {mtitle1} }} }}$",
                ),
            ],
            [masses[p1], masses[p2] / masses[p1]],
            name=f"ratio_m{mtitle1}_vs_m{mtitle2}",
        )

    for i in range(0, 4):
        ret[rf"pt_{i}"] = hmaker(
            makeAxis(60, 0, 3000, f"$p_{{T, {i}}}$", unit="GeV"),
            gj[:, i].pt,
            name=f"$p_T$ of jet {i+1}",
            description=f"$p_T$ of jet {i+1} ",
        )
        ret[f"eta_{i}"] = hmaker(
            makeAxis(20, 0, 5, f"$\eta_{{{i}}}$"),
            abs(gj[:, i].eta),
            name=f"$\eta$ of jet {i+1}",
            description=f"$\eta$ of jet {i+1}",
        )
        ret[f"phi_{i}"] = hmaker(
            makeAxis(50, 0, 4, f"$\phi_{{{i}}}$"),
            abs(gj[:, i].phi),
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

    l_bjets = events.med_bs
    ret[f"medium_bjet_pt"] = hmaker(pt_axis, l_bjets.pt, name="Medium BJet $p_{T}$")
    ret[f"medium_nb"] = hmaker(b_axis, ak.num(l_bjets.pt), name="Medium BJet Count")
    for i in range(0, 4):
        mask = ak.num(l_bjets, axis=1) > i
        ret[f"medium_b_{i}_pt"] = hmaker(
            makeAxis(20, 0, 5, f"$p_{{T}}$ of rank  {i} medium b jet"),
            l_bjets[mask][:, i].pt,
            mask=mask,
            name=f"Medium BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(l_bjets, axis=1) > 1
    top2 = l_bjets[mask]
    lb_eta = abs(top2[:, 0].eta - top2[:, 1].eta)
    lb_phi = abs(angleToNPiToPi(top2[:, 0].phi - top2[:, 1].phi))
    lb_dr = top2[:, 0].delta_r(top2[:, 1])

    ret[f"medium_bb_eta"] = hmaker(
        makeAxis(20, 0, 5, "$\\Delta \\eta$ between leading medium b jets"),
        lb_eta,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between the two highest rank medium b jets",
    )
    ret[f"medium_bb_phi"] = hmaker(
        makeAxis(25, 0, 4, "$\\Delta \\phi$ between leading medium b jets"),
        lb_phi,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between the two highest rank medium b jets",
    )
    ret[f"medium_bdr"] = hmaker(
        makeAxis(20, 0, 5, "$\\Delta R$ between leading 2 medium b jets"),
        lb_dr,
        name=rf"Medium BJet $\Delta R$",
        description=rf"$\Delta R$ between the top 2 $p_T$ b jets",
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


@analyzerModule("cr_mass_plots", ModuleType.MainHist)
def crMassHists(events, hmaker):
    ret = {}
    gj = events.good_jets

    jets = gj[:, 0:4].sum()
    mass = jets.mass

    idx = ak.local_index(gj, axis=1)
    not_med_bjet_mask = gj.btagDeepFlavB < b_tag_wps[1]
    loose_bjet_mask = gj.btagDeepFlavB > b_tag_wps[0]
    loose_not_med = not_med_bjet_mask & loose_bjet_mask
    one_lnm_mask = ak.num(loose_not_med[loose_not_med], axis=1) == 1

    mbs = events.med_bs
    sr_med_mask = ak.num(mbs, axis=1) >= 2
    sr_tight_mask = ak.num(events.tight_bs, axis=1) >= 1
    filled_med = ak.pad_none(mbs, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    dr_mask = med_dr > 1
    sr_mask = sr_med_mask & sr_tight_mask & dr_mask

    tbs = events.tight_bs
    sr_313_tight_mask = ak.num(tbs, axis=1) >= 3
    filled_tight = ak.pad_none(tbs, 2, axis=1)
    tight_dr = ak.fill_none(filled_tight[:, 0].delta_r(filled_tight[:, 1]), False)
    tight_dr_mask = tight_dr > 1
    sr_313_mask = sr_313_tight_mask & tight_dr_mask

    w = events.EventWeight
    print(sr_mask)

    ret[rf"lnm_m4_m"] = hmaker(
        makeAxis(60, 0, 3000, f"$m_{{4}}$", unit="GeV"),
        jets.mass[one_lnm_mask],
        name=rf"M4 in 1 LNM Region",
        description=rf"M4 in 1 LNM region",
        mask=one_lnm_mask,
    )

    ret[rf"sr_m4_m"] = hmaker(
        makeAxis(60, 0, 3000, f"$m_{{4}}$", unit="GeV"),
        jets.mass[sr_mask],
        name=rf"M4 in 312 SR Region",
        description=rf"M4 in 312 SR",
        mask=sr_mask,
    )

    ret[rf"sr313_m4_m"] = hmaker(
        makeAxis(60, 0, 3000, f"$m_{{4}}$", unit="GeV"),
        jets.mass[sr_313_mask],
        name=rf"M4 in 313 SR Region",
        description=rf"M4 in 313 SR",
        mask=sr_313_mask,
    )
    print(ret)

    return ret
