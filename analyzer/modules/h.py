import itertools as it

import awkward as ak
import hist
from analyzer.core import MODULE_REPO, ModuleType
import vector
import numpy as np

from .utils.btag_points import getBTagWP
from .utils.axes import makeAxis

@MODULE_REPO.register(ModuleType.Histogram)
def fatjet_plots(events, params, analyzer):
    fatjets = events.good_fatjets
    njets = ak.num(fatjets, axis=1)
    analyzer.H(
        "n_fatjets",
        makeAxis(6, -0.5, 5.5, "AK8 $N_{jets}$"),
        njets,
        description="AK8 jet multiplicity"
    )
    analyzer.H(
        "fatjet_pt",
        makeAxis(50, 140, 2000, "AK8 Jet $p_{T}$", unit="GeV"),
        fatjets.pt,
        description="pt of fatjets",
    )
    analyzer.H(
        "fatjet_eta",
        makeAxis(50, -2.5, 2.5, r"AK8 Jet $\eta$"),
        fatjets.eta,
        description="pt of fatjets",
    )
    analyzer.H(
        "fatjet_phi",
        makeAxis(50, -4, 4, r"AK8 Jet $\phi$", unit="rad"),
        fatjets.phi,
        description="pt of fatjets",
    )
    analyzer.H(
        "fatjet_msoftdrop",
        makeAxis(50, 0, 400, r"AK8 Jet Soft-Drop Mass", unit="GeV"),
        fatjets.msoftdrop,
        description="pt of fatjets",
    )
    for i in range(4):
        subji = i + 1
        analyzer.H(
            f"fatjet_{subji}subjettiness",
            makeAxis(50, 0, 1, rf"AK8 {subji}-subjettiness"),
            fatjets[f"tau{subji}"],
            description=f"{i}-subjettiness of AK8 jets"
        )
        subjj = subji - 1
        while subjj >= 1:
            analyzer.H(
            f"tau_{subji}{subjj}",
            makeAxis(50, 0, 1.3, rf"AK8 $\tau_{{{subji}{subjj}}}$"),
            fatjets[f"tau{subji}"]/fatjets[f"tau{subjj}"],
            description=f"subjettiness ratio tau_{subji}{subjj} of AK8 jets"
            )
            subjj -= 1
    fatjet_multiplicity = 3
    for i in range(fatjet_multiplicity): 
        mask = ak.num(fatjets) > i
        fatjeti = fatjets[mask][:, i]
        analyzer.H(
            f"fatjet{i}_pt",
            makeAxis(50, 140, 1000, (f"AK8 Jet {i} "+"$p_{T}$"), unit="GeV"),
            fatjeti.pt,
            description=f"pt of fatjet{i}",
            mask=mask
        )

@MODULE_REPO.register(ModuleType.Histogram)
def MET_plots(events, params, analyzer):
    met = events.MET
    analyzer.H(
        "MET_pt",
        makeAxis(50, 0, 500, r"MET $p_T$", unit="GeV"),
        met.pt,
        description=f"pt of met"
    )
    analyzer.H(
        "MET_phi",
        makeAxis(50, -4, 4, r"MET $\phi$", unit="rad"),
        met.phi,
        description=f"phi of met"
    )
    analyzer.H(
        "MET_sumEt",
        makeAxis(50, 400, 2000, r"MET $\Sigma E_{T}$", unit="GeV"),
        met.sumEt,
        description=f"sumEt of met"
    )
    analyzer.H(
        "MET_significance",
        makeAxis(50, 0, 100, "MET Significance"),
        met.significance,
        description=f"significance of met"
    )

    pmet = events.PuppiMET
    analyzer.H(
        "pMET_pt",
        makeAxis(50, 0, 500, r"PUPPI MET $p_T$", unit="GeV"),
        pmet.pt,
        description=f"pt of puppi met"
    )
    analyzer.H(
        "pMET_phi",
        makeAxis(50, -4, 4, r"PUPPI MET $\phi$", unit="rad"),
        pmet.phi,
        description=f"phi of puppi met"
    )
    analyzer.H(
        "pMET_sumEt",
        makeAxis(50, 0, 2000, r"PUPPI MET $\Sigma E_{T}$", unit="GeV"),
        pmet.sumEt,
        description=f"sumEt of puppi met"
    )
    
@MODULE_REPO.register(ModuleType.Histogram)
def jet_plots(events, params, analyzer):
    jets = events.good_jets
    njets = ak.num(jets, axis=1)
    bs = events.med_bs
    analyzer.H(
        f"jet_pt",
        makeAxis(50, 0, 2000, r"Jet $p_T$", unit="GeV"),
        jets.pt,
        description="pt of all good jets",
    )
    analyzer.H(
        f"jet_eta",
        makeAxis(50, -2.5, 2.5, r"Jet $\eta$"),
        jets.eta,
        description="eta of all good jets",
    )
    analyzer.H(
        f"jet_phi",
        makeAxis(50, -4, 4, r"Jet $\phi$", unit="rad"),
        jets.phi,
        description="phi of all good jets",
    )
    analyzer.H(
        f"jet_m",
        makeAxis(50, 0, 400, r"Jet Mass", unit="GeV"),
        jets.mass,
        description="mass of all good jets"
    )
    analyzer.H(
        f"jet_ht",
        makeAxis(50, 0, 2000, r"Jet $H_T$", unit="GeV"),
        events.HT,
        description="ht of all good jets",
    )
    analyzer.H(
        "n_jets",
        makeAxis(15, 4.5, 19.5, "AK4 $N_{jets}$"),
        njets,
        description="AK4 jet multiplicity"
    )
    analyzer.H(
        "jet_btagDeepFlavB",
        makeAxis(50, 0, 1, "Jet btagDeepFlavB Score"),
        jets.btagDeepFlavB,
        description="btag score of jets"
    )
    jet_multiplicity = 6
    for i in range(jet_multiplicity):
        jeti = jets[:, i]
        analyzer.H(
            f"jet{i}_pt",
            makeAxis(50, 0, 2000, rf"Jet {i} $p_T$", unit="GeV"),
            jeti.pt,
            description=f"pt of good jet {i}",
        )
        analyzer.H(
            f"jet{i}_eta",
            makeAxis(50, -2.5, 2.5, rf"Jet {i} $\eta$"),
            jeti.eta,
            description=f"eta of good jet {i}",
        )
        analyzer.H(
            f"jet{i}_phi",
            makeAxis(50, -4, 4, rf"Jet {i} $\phi$", unit="rad"),
            jeti.phi,
            description=f"phi of good jet {i}",
        )
        analyzer.H(
            f"jet{i}_m",
            makeAxis(50, 0, 400, rf"Jet {i} Mass", unit="GeV"),
            jeti.mass,
            description=f"mass of good jet {i}",
        )
        analyzer.H(
            f"jet{i}_btagDeepFlavB",
            makeAxis(50, 0, 1, f"Jet {i} btagDeepFlavB Score"),
            jeti.btagDeepFlavB,
            description=f"btag score of {i} jet"
        )

    # b jets
    analyzer.H(
        f"b_jet_pt",
        makeAxis(50, 0, 2000, r"b-Jet $p_T$", unit="GeV"),
        bs.pt,
        description="pt of all b jets",
    )
    analyzer.H(
        f"b_jet_eta",
        makeAxis(50, -2.5, 2.5, r"b-Jet $\eta$"),
        bs.eta,
        description="eta of all b jets",
    )
    analyzer.H(
        f"b_jet_phi",
        makeAxis(50, -4, 4, r"b-Jet $\phi$", unit="rad"),
        bs.phi,
        description="phi of all b jets",
    )
    analyzer.H(
        f"b_jet_m",
        makeAxis(50, 0, 400, r"b-Jet Mass", unit="GeV"),
        bs.mass,
        description="mass of all b jets"
    )
    for i in range(2):
        b_jeti = bs[:, i]
        analyzer.H(
            f"b_jet{i}_pt",
            makeAxis(50, 0, 2000, rf"b-Jet {i} $p_T$", unit="GeV"),
            b_jeti.pt,
            description=f"pt of b jet {i}",
        )
        analyzer.H(
            f"b_jet{i}_eta",
            makeAxis(50, -2.5, 2.5, rf"b-Jet {i} $\eta$"),
            b_jeti.eta,
            description=f"eta of b jet {i}",
        )
        analyzer.H(
            f"b_jet{i}_phi",
            makeAxis(50, -4, 4, rf"b-Jet {i} $\phi$", unit="rad"),
            b_jeti.phi,
            description=f"phi of b jet {i}",
        )
        analyzer.H(
            f"b_jet{i}_m",
            makeAxis(50, 0, 400, rf"b-Jet {i} Mass", unit="GeV"),
            b_jeti.mass,
            description=f"mass of b jet {i}",
        )
        analyzer.H(
            f"b_jet{i}_btagDeepFlavB",
            makeAxis(50, 0, 1, f"b-Jet {i} btagDeepFlavB Score"),
            b_jeti.btagDeepFlavB,
            description=f"btag score of {i} b-jet"
        )

@MODULE_REPO.register(ModuleType.Histogram)
def jetcombo_plots(events, params, analyzer):
    jets = events.good_jets
    med_bs = events.med_bs

    bwps = getBTagWP(params)
    medb_mask = jets.btagDeepFlavB > bwps["M"]
    local_indices = ak.local_index(jets, axis=1)
    toptwob_indices = local_indices[medb_mask][:,:2]
    ind_1 = toptwob_indices[:,0]
    ind_2 = toptwob_indices[:,1]
    nontop2b_mask = (local_indices != ind_1) & (local_indices != ind_2)
    nontop2b = jets[nontop2b_mask]
    non2b_fourjet = nontop2b[:,:4].sum()

    bjet1 = med_bs[:,0]
    bjet2 = med_bs[:,1]
    b_dijet = bjet1 + bjet2
    print(b_dijet['pt'])
    b_deltaR = bjet1.delta_r(bjet2)
    b_HT = ak.sum(med_bs.pt, axis=1)

    multijets = {"b_dijet": b_dijet, "non2b_fourjet": non2b_fourjet}
    for name, multijet in multijets.items():
        analyzer.H(
            f"{name}_pt",
            makeAxis(50, 0, 2000, rf"{name} $p_T$", unit="GeV"),
            multijet.pt,
            description=f"pt of {name}",
        )
        analyzer.H(
            f"{name}_eta",
            makeAxis(50, -2.5, 2.5, rf"{name} $\eta$"),
            multijet.eta,
            description=f"eta of {name}",
        )
        analyzer.H(
            f"{name}_phi",
            makeAxis(50, -4, 4, rf"{name} $\phi$", unit="rad"),
            multijet.phi,
            description=f"phi of {name}",
        )
        mrange = [0, 400] if name=="b_dijet" else [50, 1000]
        analyzer.H(
            f"{name}_m",
            makeAxis(50, mrange[0], mrange[1], rf"{name} Mass", unit="GeV"),
            multijet.mass,
            description=f"mass of {name}",
        )
    analyzer.H(
        "b_HT",
        makeAxis(50, 0, 2000, label=r"b Jet $H_T$", unit="GeV"),
        b_HT,
        description="HT of all b jets",
    )
    analyzer.H(
        "b_deltaR",
        makeAxis(50, 0, 5, label=r"b-Jet $\Delta$R"),
        b_deltaR,
        description="deltaR between top two b jets",
    )

@MODULE_REPO.register(ModuleType.Selection)
def signal_selection(events, params, selector):
    good_jets = events.good_jets
    selector.add("Lepton Veto", (ak.num(events.loose_muons) == 0) & (ak.num(events.loose_electrons) == 0))
    selector.add("2b", ak.num(events.med_bs) >= 2)
    selector.add("6jet", ak.num(good_jets) >= 6)

@MODULE_REPO.register(ModuleType.Selection)
def signal_preselection(events, params, selector):
    era_info = params.dataset.era
    bbww = era_info.trigger_names["bbww"]
    mask = events['HLT'][bbww]
    selector.add("bbWW HLT Triggers", mask)