import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType
import hist


@MODULE_REPO.register(ModuleType.Selection)
def signal_hlt(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
    selector.add(
        f"HLT_HT | HLT_AK8", events.HLT[ht_trigger_name] | events.HLT[ak8_trigger_name]
    )


@MODULE_REPO.register(ModuleType.Selection)
def hlt_HT(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    selector.add(f"HLT_HT", events.HLT[ht_trigger_name])


@MODULE_REPO.register(ModuleType.Selection)
def hlt_single_jet(events, params, selector):
    era_info = params.dataset.era
    ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
    selector.add(f"HLT_AK8", events.HLT[ak8_trigger_name])


@MODULE_REPO.register(ModuleType.Selection)
def ht_plateau(events, params, selector):
    era_info = params.dataset.era
    ht_plateau = era_info.trigger_plateaus["HT"]
    ht = events.HT
    selector.add(f"HT_Plateau", ht > ht_plateau)


@MODULE_REPO.register(ModuleType.Selection)
def softdrop_plateau(events, params, selector):
    era_info = params.dataset.era
    ak8_sd = era_info.trigger_plateaus["AK8SingleJetPt"]["msoftdrop"]

    filled_fatjets = ak.pad_none(events.good_fatjets, 1, axis=1)
    passes = ak.fill_none(filled_fatjets[:, 0].msoftdrop > ak8_sd, False)
    selector.add(f"SD_Plateau", passes)


@MODULE_REPO.register(ModuleType.Categorization)
def offline_singlejet_category(events, params, categories):
    era_info = params.dataset.era
    ak8_sd = era_info.trigger_plateaus["AK8SingleJetPt"]["msoftdrop"]
    ak8_pt = era_info.trigger_plateaus["AK8SingleJetPt"]["pt"]

    filled_fatjets = ak.pad_none(events.good_fatjets, 1, axis=1)

    first_fjet = filled_fatjets[:, 0]
    passes = ak.fill_none(
        (first_fjet.msoftdrop > ak8_sd) & (first_fjet.pt > ak8_pt), False
    )
    categories.add(
        name=f"PassOfflineSingleJet",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="PassOfflineSingleJet"
        ),
        values=passes,
    )


@MODULE_REPO.register(ModuleType.Categorization)
def offline_ht_category(events, params, categories):
    era_info = params.dataset.era
    ht_plateau = era_info.trigger_plateaus["HT"]
    ht = events.HT
    categories.add(
        name=f"PassOfflineHT",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="PassOfflineHT"
        ),
        values=ht > ht_plateau,
    )


@MODULE_REPO.register(ModuleType.Categorization)
def hlt_ht_trigger_category(events, params, categories):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    categories.add(
        name=f"PassHLTHT",
        axis=hist.axis.Integer(0, 2, underflow=False, overflow=False, name="PassHLTHT"),
        values=events.HLT[ht_trigger_name],
    )


@MODULE_REPO.register(ModuleType.Categorization)
def hlt_singlejet_trigger_category(events, params, categories):
    era_info = params.dataset.era
    ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
    categories.add(
        name=f"PassHLTSingleJet",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="PassHLTSingleJet"
        ),
        values=events.HLT[ak8_trigger_name],
    )


@MODULE_REPO.register(ModuleType.Selection)
def single_muon(events, params, selector):
    era_info = params.dataset.era
    single_muon_trigger_name = era_info.trigger_names["SingleMuon"]
    iso_muon_trigger_name = era_info.trigger_names["IsoMuon"]
    selector.add(
        f"HLT_SingleMu | HLT_IsoMu",
        events.HLT[single_muon_trigger_name] | events.HLT[iso_muon_trigger_name],
    )


@MODULE_REPO.register(ModuleType.Selection)
def general_selection(events, params, selector):
    """Signal selection without b cuts"""
    good_jets = events.good_jets
    good_muons = events.good_muons
    good_electrons = events.good_electrons
    filled_jets = ak.pad_none(good_jets, 4, axis=1)
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    passes_highptjet = ak.fill_none(filled_jets[:, 0].pt > 300, False)
    selector.add("highptjet", passes_highptjet)

    passes_jets = (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 6)
    selector.add("njets", passes_jets)

    passes_0Lep = (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0)
    selector.add("0Lep", passes_0Lep)


@MODULE_REPO.register(ModuleType.Selection)
def partial_signal312_selection(events, params, selector):
    """Signal selection without b cuts"""
    med_b = events.med_bs
    tight_b = events.tight_bs
    filled_med = ak.pad_none(med_b, 2, axis=1)
    med_dr = ak.fill_none(filled_med[:, 0].delta_r(filled_med[:, 1]), False)
    passes_2bjet = ak.num(med_b) >= 2
    passes_1tightbjet = ak.num(tight_b) >= 1
    passes_b_dr = med_dr > 1
    selector.add("2bjet", passes_2bjet)
    selector.add("1tightbjet", passes_1tightbjet)
    selector.add("b_dr", passes_b_dr)


@MODULE_REPO.register(ModuleType.Selection)
def partial_signal313_selection(events, params, selector):
    """Signal 313 selection without b cuts"""
    tight_b = events.tight_bs
    filled_tight = ak.pad_none(tight_b, 2, axis=1)
    tight_dr = ak.fill_none(filled_tight[:, 0].delta_r(filled_tight[:, 1]), False)
    passes_3tightbjet = ak.num(tight_b) >= 3
    passes_b_dr = tight_dr > 1
    selector.add("3tightbjet", passes_3tightbjet)
    selector.add("b_dr", passes_b_dr)


@MODULE_REPO.register(ModuleType.Selection)
def partial_cr_selection(events, params, selector):
    """Control region selection.
    Requires 0 loose bs.
    """
    loose_b = events.loose_bs
    selector.add("0looseb", (ak.num(loose_b) == 0))


@MODULE_REPO.register(ModuleType.Selection)
def dijet_selection(events, params, selector):
    """Signal selection for dijet analysis"""
    good_jets = events.good_jets
    wide_jet0 = events.wide_jet0
    wide_jet1 = events.wide_jet1
    #fat_jets = events.fat_jets 
    #electrons = events.good_electrons
    #muons = events.good_muons

    d_eta = abs(wide_jet0.eta - wide_jet1.eta)
    dijet = wide_jet0 + wide_jet1

    passes_njets = ak.fill_none((ak.num(good_jets) >= 2), False)
    selector.add("njets", passes_njets)

    passes_dijet_mass = (dijet.mass > 1530)
    selector.add("dijet_mass", passes_dijet_mass)

    passes_dijet_eta = ak.fill_none((d_eta < 1.1), False)
    selector.add("dijet_eta", passes_dijet_eta)

    #filled_fatjets = ak.pad_none(fat_jets, 2, axis=1)
    #passes_fatjet_mass = ak.fill_none(((filled_fatjets[:, 0].msoftdrop < 65) & (filled_fatjets[:, 1].msoftdrop < 65)), False)
    #selector.add("fatjet_mass", passes_fatjet_mass)

    #near_el = ak.any(~ak.is_none(filled_jets[:,0:2].nearest(electrons, threshold=0.4),axis=1),axis=1)
    #near_mu = ak.any(~ak.is_none(filled_jets[:,0:2].nearest(muons, threshold=0.4),axis=1),axis=1)

    #ak.num(near_el is not None)
    #no_electrons = (ak.num(ak.drop_none(good_jets.nearest(electrons,threshold=0.4))) == 0)
    #no_muons = (ak.num(ak.drop_none(good_jets.nearest(muons,threshold=0.4))) == 0)

    #no_electrons = (ak.num(electrons) == 0)
    #no_muons = (ak.num(muons) == 0)

    #no_leptons = no_electrons & no_muons
    #selector.add("el_veto", ~(near_el))
    #selector.add("mu_veto", ~(near_mu))


@MODULE_REPO.register(ModuleType.Categorization)
def one_btag_category(events, params, categories):
    """Categorization for events with one b-tagged jet"""
    med_b = events.medium_bs
    passes_one_btag = (ak.num(med_b) == 1)

    categories.add(
        name="OneBTag",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="OneBTag"
        ),
        values=passes_one_btag,
    )

@MODULE_REPO.register(ModuleType.Categorization)
def two_btag_category(events, params, categories):
    """Categorization for events with two b-tagged jets"""
    med_b = events.medium_bs
    passes_two_btag = (ak.num(med_b) == 2)
    categories.add(
        name="TwoBTag",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="TwoBTag"
        ),
        values=passes_two_btag,
    )
@MODULE_REPO.register(ModuleType.Categorization)
def zero_btag_category(events, params, categories):
    """Categorization for events with no b-tagged jets, but with one muon"""
    med_b = events.medium_bs
    good_muons = events.good_muons
    passes_zero_btag = (ak.num(med_b) == 0) & ((ak.num(good_muons[good_muons.looseId]) >= 1))
    categories.add(
        name="OneMu",
        axis=hist.axis.Integer(
            0, 2, underflow=False, overflow=False, name="OneMu"
        ),
        values=passes_zero_btag,
    )

@MODULE_REPO.register(ModuleType.Selection)
def signal_dijet_hlt(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    ak8_trigger_name = era_info.trigger_names["AK8SingleJetPt"]
    pf_trigger_name = era_info.trigger_names["PFSingleJetPt"]
    calo_trigger_name = era_info.trigger_names["CaloSingleJetPt"]
    selector.add(
        f"HLT_HT | HLT_AK8 | HLT_PFJet | HLT_CaloJet",
        events.HLT[ht_trigger_name]
        | events.HLT[ak8_trigger_name]
        | events.HLT[pf_trigger_name]
        | events.HLT[calo_trigger_name],
    )
