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

