import itertools as it

import awkward as ak

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi

from .axes import *
from .objects import b_tag_wps
from .utils import numMatching


@analyzerModule("pre_sel_hists", categories="presel")
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


@analyzerModule("event_level_hists", categories="main", depends_on=["objects"])
def createEventLevelHistograms(events, analyzer):
    ret = {}
    analyzer.H(
        f"HT",
        makeAxis(60, 0, 3000, "HT", unit="GeV"),
        events.HT,
        name="Event HT",
        description="Sum of $p_T$ of good AK4 jets.",
    )
    if "LHE" not in events.fields:
        return events, analyzer
    analyzer.H(
        f"nQLHE",
        makeAxis(10, 0, 10, "Quark Count LHE"),
        events.LHE.Nuds + events.LHE.Nc + events.LHE.Nb,
        name="Quark Count LHE",
        description="Number of LHE level Quarks",
    )
    analyzer.H(
        f"nJLHE",
        makeAxis(10, 0, 10, "Jet Count LHE"),
        events.LHE.Njets,
        name="Jet Count LHE",
        description="Number of LHE level Jets",
    )
    analyzer.H(
        f"nGLHE",
        makeAxis(10, 0, 10, "Gluon Count LHE"),
        events.LHE.Nglu,
        name="Gluon Count LHE",
        description="Number of LHE level gluons",
    )
    return events, analyzer


<<<<<<< HEAD
@analyzerModule("tag_hists", depends_on=["objects"])
=======
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


=======
@analyzerModule("tag_hists", ModuleType.MainHist)
>>>>>>> parent of 091d9c5... mSoftDrop trigger efficiency
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
