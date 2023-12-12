from analyzer.core import analyzerModule, ModuleType
from analyzer.math_funcs import angleToNPiToPi
from .axes import *
import awkward as ak
from .objects import b_tag_wps
import itertools as it
from .utils import numMatching


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
        mask = ak.num(gj, axis=1) > i
        gj_mask = gj[mask]
        eta = gj_mask[:, i].eta
        phi = gj_mask[:, i].phi

        ret[rf"pt_{i}"] = hmaker(
            pt_axis,
            gj_mask[:, i].pt,
						mask = mask,
            name=f"$p_T$ of jet {i+1}",
            description=f"$p_T$ of jet {i+1} ",
        )
        ret[f"eta_{i}"] = hmaker(
            eta_axis,
            eta,
						mask = mask,
            name=f"$\eta$ of jet {i+1}",
            description=f"$\eta$ of jet {i+1}",
        )
        ret[f"phi_{i}"] = hmaker(
            phi_axis,
            phi,
						mask = mask,
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

