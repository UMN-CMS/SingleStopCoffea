import awkward as ak
from analyzer.core import analyzerModule
import functools
import operator as op
from .axes import *


@analyzerModule(
    "baseline_selection",
    categories="selection",
    depends_on=["objects"],
    processing_info={"used_btag_wps": ["M", "T"]},
)
def createSelection(events, analyzer):
    """Baseline selection for the analysis.
    Applies the following selection:
    - Jets[0].pt > 300
    - 4 <= nJets <= 6
    - 0 leptons
    - 2 medium bjets, at least one of which is tight
    - delta_R(med_bjets[0],med_bjets[1]) > 1
    """

    selection = analyzer.selection
    good_jets = events.good_jets
    fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    med_b = events.med_bs
    tight_b = events.tight_bs
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
    filled_med = ak.pad_none(med_b, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    hlt_names = analyzer.profile.hlt
    passes_highptjet = ak.fill_none(filled_jets[:, 0].pt > 300, False)
    passes_jets = (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)
    passes_0Lep = (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)
    passes_2bjet = ak.num(med_b) >= 2
    passes_1tightbjet = ak.num(tight_b) >= 1
    passes_b_dr = med_dr > 1

    if "HLT" in events.fields:
        hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
        #passes_hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
        selection.add("hlt", hlt)
        #for n in hlt_names:
        #hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
            #selection.add(f"hlt_{n}", events.HLT[n])

    selection.add("highptjet", passes_highptjet)
    selection.add("jets", passes_jets)
    selection.add("0Lep", passes_0Lep)
    selection.add("2bjet", passes_2bjet)
    selection.add("1tightbjet", passes_1tightbjet)
    selection.add("b_dr", passes_b_dr)

    return events, analyzer

@analyzerModule(
    "baseline_muon_selection",
    categories="selection",
    depends_on=["objects"],
    processing_info={"used_btag_wps": ["M", "T"]},
)
def createMuonSelection(events, analyzer):
    selection = analyzer.selection
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    hlt_names = analyzer.profile.hlt
    passes_0Electron = (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 1)

    if "HLT" in events.fields:
        for n in hlt_names:
            if 'HT' not in n and 'AK8PFJet' not in n and 'Mu' in n: selection.add(f"hlt_{n}", events.HLT[n])
    selection.add("0Electron", passes_0Electron)
    
    return events, analyzer

@analyzerModule("trigger_efficiency_hists", categories="main")
def triggerEfficiencyHists(events, analyzer):
    gj = events.good_jets
    nmj = events.non_muonic_jets
    fatjets = events.FatJet
    fatjets = fatjets[(events.FatJet.pt > 175) & (abs(events.FatJet.eta) < 2.4)]
    hlt_names = analyzer.profile.hlt
    if "HLT" in events.fields:
        for n in hlt_names:
            if 'HT' in n: HT_Trigger = events.HLT[n]
            if 'AK8PFJet' in n: pT_Trigger = events.HLT[n]
    
    mht = HT_Trigger
    mpt = (pT_Trigger & (ak.num(fatjets, axis=1) >= 1))
    mnj = ak.num(fatjets, axis = 1) >= 1
    analyzer.H(
        "passedHT",
        makeAxis(100, 0, 2000, "HT", unit="GeV"),
        events[mht].NonMuonHT,
        mask = mht,
    ) 

    analyzer.H(
        "totalHT",
        makeAxis(100, 0, 2000, "HT", unit="GeV"),
        events.NonMuonHT,
    )

    analyzer.H(
        "passed_pt0",
        makeAxis(30, 0, 1500, "$p_{T, 1}$", unit="GeV"),
        fatjets[mpt][:, 0].pt,
        mask = mpt,
    )

    analyzer.H(
        "total_pt0",
        makeAxis(30, 0, 1500, "$p_{T, 1}$", unit="GeV"),
        fatjets[mnj][:, 0].pt,
        mask = mnj,
    )

    analyzer.H(
        "total_pT0_vs_mSoftDrop",
        [ makeAxis(30, 0, 1500, "$p_{T, 1}$", unit="GeV"),
          makeAxis(50, 0, 500, "$m_{SD, 1}$", unit="GeV")
        ],
        [fatjets[mnj][:, 0].pt, fatjets[mnj][:, 0].msoftdrop],
        mask = mnj,
    )

    analyzer.H(
        "passed_pT0_vs_mSoftDrop",
        [ makeAxis(30, 0, 1500, "$p_{T, 1}$", unit="GeV"),
          makeAxis(50, 0, 500, "$m_{SD, 1}$", unit="GeV")
        ],
        [fatjets[mpt][:, 0].pt, fatjets[mpt][:, 0].msoftdrop],
        mask = mpt,
    )

    j = 1
    mnj = (ak.num(nmj, axis = 1) >= j) 
    masked_jets = nmj[mnj]
    muons = events.good_muons[mnj]
    analyzer.H(
        "d_r_mu_j{0}".format(j),
        makeAxis(25, 0, 5, "$|\Delta R_{{{\mu}{j_1}}}|$"),
        abs(masked_jets[:, j - 1].delta_r(muons[:, 0])),
        mask = mnj,
    )

    j = 2
    mnj = (ak.num(nmj, axis = 1) >= j) 
    masked_jets = nmj[mnj]
    muons = events.good_muons[mnj]
    analyzer.H(
        "d_r_mu_j{0}".format(j),
        makeAxis(25, 0, 5, "$|\Delta R_{{{\mu}{j_2}}}|$"),
        abs(masked_jets[:, j - 1].delta_r(muons[:, 0])),
        mask = mnj,
    )

    j = 3
    mnj = (ak.num(nmj, axis = 1) >= j) 
    masked_jets = nmj[mnj]
    muons = events.good_muons[mnj]
    analyzer.H(
        "d_r_mu_j{0}".format(j),
        makeAxis(25, 0, 5, "$|\Delta R_{{{\mu}{j_3}}}|$"),
        abs(masked_jets[:, j - 1].delta_r(muons[:, 0])),
        mask = mnj,
    )

    j = 4
    mnj = (ak.num(nmj, axis = 1) >= j) 
    masked_jets = nmj[mnj]
    muons = events.good_muons[mnj]
    analyzer.H(
        "d_r_mu_j{0}".format(j),
        makeAxis(25, 0, 5, "$|\Delta R_{{{\mu}{j_4}}}|$"),
        abs(masked_jets[:, j - 1].delta_r(muons[:, 0])),
        mask = mnj,
    )

    return events, analyzer

@analyzerModule("baseline_hists", categories="main")
def selectionHists(events, analyzer):
    gj = events.good_jets
    nj = ak.num(gj, axis=1)
    analyzer.H("njets", makeAxis(10, 0, 10, f"NJets"), nj)

    mnj = nj >= 1
    analyzer.H(
        "pt0",
        makeAxis(100, 0, 1500, "$p_{T,0}$", unit="GeV"),
        gj[mnj][:, 0].pt,
        mask=mnj,
    )

    good_muons = events.good_muons
    good_electrons = events.good_electrons

    analyzer.H(
        "nlep", makeAxis(6, 0, 6, f"NLep"), ak.num(good_electrons) + ak.num(good_muons)
    )

    med_b = events.med_bs
    tight_b = events.tight_bs
    loose_b = events.loose_bs

    nb = ak.num(med_b, axis=1)

    analyzer.H("n_medb", makeAxis(6, 0, 6, f"N Medium b"), nb)
    analyzer.H("n_tightb", makeAxis(6, 0, 6, f"N Tight b"), ak.num(tight_b))
    analyzer.H("n_looseb", makeAxis(6, 0, 6, f"N Loose b"), ak.num(loose_b))

    mnb = nb >= 2
    twob = med_b[mnb]
    med_dr = twob[:, 0].delta_r(twob[:, 1])
    analyzer.H("med_b_dr", makeAxis(20, 0, 5, "$\Delta R$"), med_dr, mask=mnb)

    analyzer.H(
        "HT",
        makeAxis(60, 0, 3000, "HT", unit="GeV"),
        events.HT,
        name="Event HT",
        description="Sum of $p_T$ of good AK4 jets.",
    )

    analyzer.H(f"phi_vs_eta",
                [makeAxis(50,-5.0,5.0,f"$\eta$"), makeAxis(50,-5.0,5.0,f"$\phi$")],
                [gj.eta, gj.phi],
                name=f"$\eta$ vs $\phi$ of jet ",
                description=rf"$\eta$ vs $\phi$ of jet "
                )

    return events, analyzer


@analyzerModule(
    "baseline_nodr",
    categories="selection",
    depends_on=["objects"],
    processing_info={"used_btag_wps": ["M", "T"]},
)
def baselineNoDR(events, analyzer):
    """Baseline selection for the analysis.
    Applies the following selection:
    - Jets[0].pt > 300
    - 4 <= nJets <= 6
    - 0 leptons
    - 2 medium bjets, at least one of which is tight
    """

    selection = analyzer.selection
    good_jets = events.good_jets
    # fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    # loose_b = events.loose_bs
    med_b = events.med_bs
    tight_b = events.tight_bs
    # tight_top = events.tight_tops
    # selection = PackedSelection()
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    # top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    filled_med = ak.pad_none(med_b, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    # selection.add("trigger", (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6))
    hlt_names = analyzer.profile.hlt
    if "HLT" in events.fields:
        hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
        selection.add("hlt", hlt)
    selection.add("highptjet", (ak.fill_none(filled_jets[:, 0].pt > 300, False)))
    selection.add("jets", ((ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)))
    selection.add("0Lep", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)))
    selection.add("2bjet", (ak.num(med_b) >= 2))
    selection.add("1tightbjet", (ak.num(tight_b) >= 1))
    # selection.add("jet_dr", ((top_two_dr < 4) & (top_two_dr > 2)).to_numpy())
    return events, analyzer


@analyzerModule("bbpt_selection", categories="selection", depends_on=["objects"])
def createBBptSelection(events, analyzer):
    selection = analyzer.selection
    med_b = events.med_bs
    filled_med = ak.pad_none(med_b, 2, axis=1)
    bbpt = ak.fill_none((filled_med[:, 0] + filled_med[:, 1]).pt, False)
    selection.add("bbpt", (bbpt > 200).to_numpy())
    return events, analyzer


@analyzerModule(
    "cr_selection",
    categories="selection",
    depends_on=["objects"],
    processing_info={
        "used_btag_wps": ["L"],
    },
)
def createCRSelection(events, analyzer):
    selection = analyzer.selection
    good_jets = events.good_jets
    # fat_jets = events.FatJet
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    loose_b = events.loose_bs
    # med_b = events.med_bs
    # tight_b = events.tight_bs
    # tight_top = events.tight_tops
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    # top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)
    hlt_names = analyzer.profile.hlt
    if "HLT" in events.fields:
        for n in hlt_names:
        #hlt = functools.reduce(op.or_, [events.HLT[x] for x in hlt_names])
            selection.add(f"hlt_{n}", events.HLT[n])
    selection.add("ht1200", (events.HT >= 1200))
    selection.add("highptjet", (ak.fill_none(filled_jets[:, 0].pt > 300, False)))
    selection.add("jets", ((ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)))
    selection.add("0Lep", ((ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)))
    selection.add("0looseb", (ak.num(loose_b) == 0))

    #analyzer.critical_selections.append("0looseb")

    return events, analyzer
