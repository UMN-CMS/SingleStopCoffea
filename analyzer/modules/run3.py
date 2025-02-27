import awkward as ak
from analyzer.core import analyzerModule
import functools
import operator as op
from .axes import *
from .btag_points import getBTagWP

def makeCutSet(x, s, args):
    return [x[s > a] for a in args]

@analyzerModule("run3_selection", categories="selection")
def run3Selection(events, analyzer):
    good_jets = events.Jet[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)]
    fat_jets = events.FatJet[(events.FatJet.pt > 175) & (abs(events.FatJet.eta) < 2.4)]
    bwps = getBTagWP(analyzer.profile)

    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
       [bwps["L"], bwps["M"], bwps["T"]],
    )

    HT = ak.sum(good_jets.pt, axis = 1)

    selection = analyzer.selection
    trig = ((HT > 280) & (ak.num(good_jets, axis = 1) >= 4) & (ak.num(med_b, axis = 1) >= 2)) 
    selection.add("Simulated trigger", trig)
    return events, analyzer

@analyzerModule("run3_hists", categories="main")
def run3Hists(events, analyzer):
    trig = ((events.HT > 280) & (ak.num(events.good_jets, axis = 1) >= 4) & (ak.num(events.med_bs, axis = 1) >= 2)) 
    gj = events.good_jets[trig]
    med_bs = events.med_bs[trig]
    tight_bs = events.tight_bs[trig]

    analyzer.H(
        "m3",
        mass_axis,
        gj[:,0:3].sum().mass,
        mask = trig,
    )

    analyzer.H(
        "m4",
        mass_axis,
        gj[:, 0:4].sum().mass,
        mask = trig,
    )

    analyzer.H(
        "ratio_m3_m4",
        [ makeAxis(50, 0, 500, "$m_{3}$", unit="GeV"),
          makeAxis(50, 0, 500, "$m_{4}$", unit="GeV")
        ],
        [gj[:, 0:3].sum().mass, gj[:, 0:4].sum().mass],
        mask = trig,
    )

    analyzer.H(
        "pt_0",
        pt_axis,
        gj[:,0].pt,
        mask = trig,
    )

    analyzer.H(
        "HT",
        ht_axis,
        events[trig].HT,
        mask = trig,
    )

    analyzer.H(
        "nb_med",
        b_axis,
        ak.num(med_bs, axis=1),
        mask = trig,
    )     

    analyzer.H(
        "nb_tight",
        b_axis,
        ak.num(tight_bs, axis=1),
        mask = trig,
    )     

    analyzer.H(
        "nj",
        nj_axis,
        ak.num(gj, axis=1),
        mask = trig,
    )

    analyzer.H(
        "dRbb_01",
        dr_axis,
        abs(med_bs[:, 0].delta_r(med_bs[:, 1])),
        mask = trig,
    )

    return events, analyzer

@analyzerModule("run3_signal_hists", categories="main")
def triggerEfficiencyHists(events, analyzer):
    selection = analyzer.selection
    gj = events.good_jets
    
    hlt_names = analyzer.profile.hlt
    if "HLT" in events.fields:
        for n in hlt_names:
            if 'HT' in n and 'QuadPFJet' not in n: HT_Trigger = events.HLT[n]
            if 'AK8PFJet' in n: pT_Trigger = events.HLT[n]
            if 'QuadPFJet' in n: b_Trigger = events.HLT[n]


    trig = ((events.HT > 280) & (ak.num(gj, axis = 1) >= 4) & (ak.num(events.med_bs, axis = 1) >= 2)) 
    selection.add(trig)

    
    return events, analyzer
