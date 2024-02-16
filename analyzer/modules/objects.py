import awkward as ak
from analyzer.core import analyzerModule


def makeCutSet(x, s, *args):
    return [x[s > a] for a in args]


MCCampaign = "UL2018"
if MCCampaign == "UL2016preVFP":
    b_tag_wps = [0.0508, 0.2598, 0.6502]
elif MCCampaign == "UL2016postVFP":
    b_tag_wps = [0.0480, 0.2489, 0.6377]
elif MCCampaign == "UL2017":
    b_tag_wps = [0.0532, 0.3040, 0.7476]
elif MCCampaign == "UL2018":
    b_tag_wps = [0.0490, 0.2783, 0.7100]


@analyzerModule("objects", categories="base_objects")
def createObjects(events, analyzer):
    good_jets = events.Jet[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)]
    fat_jets = events.FatJet[(events.FatJet.pt > 30) & (abs(events.FatJet.eta) < 2.4)]
    loose_top, med_top, tight_top = makeCutSet(
        fat_jets, fat_jets.particleNet_TvsQCD, 0.58, 0.80, 0.97
    )
    #loose_W, med_W, tight_W = makeCutSet(
    #    fat_jets, fat_jets.particleNet_WvsQCD, 0.7, 0.94, 0.98
    #)

    #deep_top_wp1, deep_top_wp2, deep_top_wp3, deep_top_wp4 = makeCutSet(
    #    fat_jets, fat_jets.deepTag_TvsQCD, 0.436, 0.802, 0.922, 0.989
    #)
    #deep_W_wp1, deep_W_wp2, deep_W_wp3, deep_W_wp4 = makeCutSet(
    #    fat_jets, fat_jets.deepTag_WvsQCD, 0.458, 0.762, 0.918, 0.961
    #)
    loose_b, med_b, tight_b = makeCutSet(
        good_jets, good_jets.btagDeepFlavB, *(b_tag_wps[x] for x in range(3))
    )

    el = events.Electron
    good_electrons = el[
        (el.cutBased == 4)
        & (el.miniPFRelIso_all < 0.1)
        & (el.pt > 30)
        & (abs(el.eta) < 2.4)
    ]
    mu = events.Muon
    good_muons = mu[
        (mu.mediumId) & (mu.miniPFRelIso_all < 0.2) & (mu.pt > 30) & (abs(mu.eta) < 2.4)
    ]
    events["good_jets"] = good_jets
    events["good_electrons"] = good_electrons
    events["good_muons"] = good_muons

    events["loose_bs"] = loose_b
    events["med_bs"] = med_b
    events["tight_bs"] = tight_b

    events["tight_tops"] = tight_top
    # events["med_tops"] = med_top
    # events["loose_tops"] = loose_top

    # events["tight_Ws"] = tight_W
    # events["med_Ws"] = med_W
    # events["loose_Ws"] = loose_W

    # events["deep_top_wp1"] = deep_top_wp1
    # events["deep_top_wp2"] = deep_top_wp2
    # events["deep_top_wp3"] = deep_top_wp3
    # events["deep_top_wp4"] = deep_top_wp4

    # events["deep_W_wp1"] = deep_W_wp1
    # events["deep_W_wp2"] = deep_W_wp2
    # events["deep_W_wp3"] = deep_W_wp3
    # events["deep_W_wp4"] = deep_W_wp4

    return events, analyzer
