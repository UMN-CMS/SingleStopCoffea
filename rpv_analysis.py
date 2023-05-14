from pathlib import Path
from coffea.nanoevents import NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea import processor
import coffea as cf
import pickle
import awkward as ak
from framework.execution import executeConfiguration
from framework.basic_weights import getFileWeight
from framework.skimming import uproot_writeable
from coffea.processor import accumulate
import framework.fileutils as futil
import os
import uproot
import hist
import itertools
from collections import namedtuple
import collections.abc as cabc


import yaml

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper

import logging
import logging.config

log_conf = yaml.load(open("logconf.yaml", "r"), Loader=Loader)
logging.config.dictConfig(log_conf)
a_logger = logging.getLogger("AnalysisLogger")

dataset_axis = hist.axis.StrCategory(
    [], growth=True, name="dataset", label="Primary dataset"
)

mass_axis = hist.axis.Regular(150, 0, 3000, name="mass", label=r"$m$ [GeV]")
pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")
ht_axis = hist.axis.Regular(300, 0, 3000, name="ht", label=r"$HT$ [GeV]")
dr_axis = hist.axis.Regular(20, 0, 5, name="dr", label=r"$\Delta R$")
eta_axis = hist.axis.Regular(20, -5, 5, name="eta", label=r"$\eta$")
phi_axis = hist.axis.Regular(50, 0, 4, name="phi", label=r"$\phi$")
nj_axis = hist.axis.Regular(10, 0, 10, name="nj", label=r"$n_{j}$")
tencountaxis = hist.axis.Regular(10, 0, 10, name="Number", label=r"Number")
b_axis = hist.axis.Regular(5, 0, 5, name="nb", label=r"$n_{b}$")


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


def createObjects(events):
    good_jets = events.Jet[(events.Jet.pt > 30) & (abs(events.Jet.eta) < 2.4)]
    fat_jets = events.FatJet
    loose_top, med_top, tight_top = makeCutSet(
        fat_jets, fat_jets.particleNet_TvsQCD, 0.58, 0.80, 0.97
    )
    loose_W, med_W, tight_W = makeCutSet(
        fat_jets, fat_jets.particleNet_WvsQCD, 0.7, 0.94, 0.98
    )

    deep_top_wp1, deep_top_wp2, deep_top_wp3, deep_top_wp4 = makeCutSet(
        fat_jets, fat_jets.deepTag_TvsQCD, 0.436, 0.802, 0.922, 0.989
    )
    deep_W_wp1, deep_W_wp2, deep_W_wp3, deep_W_wp4 = makeCutSet(
        fat_jets, fat_jets.deepTag_WvsQCD, 0.458, 0.762, 0.918, 0.961
    )
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

    events["loose_bs"] = loose_b
    events["med_bs"] = med_b
    events["tight_bs"] = tight_b

    events["tight_tops"] = tight_top
    events["med_tops"] = med_top
    events["loose_tops"] = loose_top

    events["tight_Ws"] = tight_W
    events["med_Ws"] = med_W
    events["loose_Ws"] = loose_W

    events["deep_top_wp1"] = deep_top_wp1
    events["deep_top_wp2"] = deep_top_wp2
    events["deep_top_wp3"] = deep_top_wp3
    events["deep_top_wp4"] = deep_top_wp4

    events["deep_W_wp1"] = deep_W_wp1
    events["deep_W_wp2"] = deep_W_wp2
    events["deep_W_wp3"] = deep_W_wp3
    events["deep_W_wp4"] = deep_W_wp4

    return events, {}


def isGoodGenParticle(particle):
    return particle.hasFlags("isLastCopy", "fromHardProcess") & ~(
        particle.hasFlags("fromHardProcessBeforeFSR")
        & ((abs(particle.pdgId) == 1) | (abs(particle.pdgId) == 3))
    )


ParticleChain = namedtuple("ParticleChain", "name pdgid_list children")

structure = [
    ParticleChain(
        "Stop",
        1000006,
        [
            ParticleChain("BStop", 5, None),
            ParticleChain(
                "Chi",
                1000024,
                [
                    ParticleChain("BChi", 5, None),
                    ParticleChain("SChi", 3, None),
                    ParticleChain("DChi", 1, None),
                ],
            ),
        ],
    )
]


def genMatchParticles(children, structure, allow_anti=True):
    ret = {}
    for s_child in structure:
        found = children[
            s_child.pdgid == abs(children.pdgid) if allow_anti else children.pdgid
        ]
        if ak.any(ak.num(found, axis=1) != 1):
            raise ValueError("AAAAAAAAAAAAA")
        ret[s_child.name] = found
        if structure.children:
            ret.update(_genMatchParticles(found, structure.children))
    return ret


def goodGenParticles(events):
    events["good_gen_particles"] = events.GenPart[isGoodGenParticle(events.GenPart)]
    gg = events.good_gen_particles
    test = genMatchParticles(gg, structure)
    return events


def createSelection(events):
    a_logger.debug(f"Creating Selection")
    good_jets = events.good_jets
    fat_jets = events.FatJet

    good_muons = events.good_muons
    good_electrons = events.good_electrons

    loose_b = events.loose_bs
    tight_top = events.tight_tops

    selection = PackedSelection()

    filled_jets = ak.pad_none(good_jets, 2, axis=1)
    # filled_jets = good_jets
    top_two_dr = ak.fill_none(filled_jets[:, 0].delta_r(filled_jets[:, 1]), False)

    selection.add("jets", (ak.num(good_jets) >= 4) & (ak.num(good_jets) <= 5))
    selection.add("0Lep", (ak.num(good_electrons) == 0) & (ak.num(good_muons) == 0))
    selection.add("2bjet", ak.num(loose_b) >= 2)
    selection.add("highptjet", ak.fill_none(filled_jets[:, 0].pt > 300, False))
    selection.add("jet_dr", (top_two_dr < 4) & (top_two_dr > 2))
    # selection.add("0Top", ak.num(tight_top) == 0)
    return selection


def makeHistogram(
    axis, dataset, data, weights, name=None, description=None, drop_none=True
):
    if isinstance(axis, list):
        h = hist.Hist(dataset_axis, *axis, storage="weight", name=name)
    else:
        h = hist.Hist(dataset_axis, axis, storage="weight", name=name)
    setattr(h, "description", description)
    if isinstance(axis, list):
        ret = h.fill(dataset, *data, weight=weights)
    else:
        ret = h.fill(dataset, ak.to_numpy(data), weight=weights)
    return ret


def createJetHistograms(events):
    ret = {}
    dataset = events.metadata["dataset"]
    gj = events.good_jets
    w = events.EventWeight

    ret[f"h_njet"] = makeHistogram(nj_axis, dataset, ak.num(gj), w)

    for i, j in [(0, 3), (1, 4), (0, 4)]:
        jets = gj[:, i:j].sum()
        ret[f"m{i}{j}_pt"] = makeHistogram(
            pt_axis,
            dataset,
            jets.pt,
            w,
            name=f"Composite Jet {i} to Jet {j} $p_T$",
            description=f"$p_T$ of the sum of jets {i} to {j}",
        )
        ret[f"m{i}{j}_eta"] = makeHistogram(
            eta_axis,
            dataset,
            jets.eta,
            w,
            name=f"Composite Jet {i} to Jet {j} $\eta$",
            description=f"$\eta$ of the sum of jets {i} to {j}",
        )
        ret[f"m{i}{j}_m"] = makeHistogram(
            mass_axis,
            dataset,
            jets.mass,
            w,
            name=f"Composite Jet {i} to Jet {j} mass",
            description=f"Mass of the sum of jets {i} to {j}",
        )
    for i in range(0, 4):
        ret[f"pt_{i}"] = makeHistogram(
            pt_axis,
            dataset,
            gj[:, i].pt,
            w,
            name=f"$p_T$ of jet {i}",
            description=f"$p_T$ of jet {i} (indexed from 0)",
        )
        ret[f"eta_{i}"] = makeHistogram(
            eta_axis,
            dataset,
            gj[:, i].eta,
            w,
            name=f"$\eta$ of jet {i}",
            description=f"$\eta$ of jet {i}(indexed from 0)",
        )
        ret[f"phi_{i}"] = makeHistogram(
            phi_axis,
            dataset,
            gj[:, i].phi,
            w,
            name=f"$\phi$ of jet {i}",
            description=f"$\phi$ of jet {i}(indexed from 0)",
        )

    padded_jets = ak.pad_none(gj, 5, axis=1)
    masks = {}
    for i, j in list(x for x in itertools.combinations(range(0, 5), 2) if x[0] != x[1]):
        mask = ak.num(gj, axis=1) > max(i, j)
        w_mask = w[mask]
        masked_jets = gj[mask]
        d_eta = masked_jets[:, i].eta - masked_jets[:, j].eta
        d_r = masked_jets[:, i].delta_r(masked_jets[:, j])
        d_phi = masked_jets[:, i].phi - masked_jets[:, j].phi
        masks[(i, j)] = mask
        ret[f"d_eta_{i}_{j}"] = makeHistogram(
            eta_axis,
            dataset,
            d_eta,
            w_mask,
            name=f"$\Delta \eta$ between jets {i} and {j}",
            description=f"$\Delta \eta$ between jets {i} and {j}, indexed from 0",
        )
        ret[f"d_phi_{i}_{j}"] = makeHistogram(
            phi_axis,
            dataset,
            d_phi,
            w_mask,
            name=f"$\Delta \phi$ between jets {i} and {j}",
            description=f"$\Delta \phi$ between jets {i} and {j}, indexed from 0",
        )
        ret[f"d_r_{i}_{j}"] = makeHistogram(
            dr_axis,
            dataset,
            d_r,
            w_mask,
            name=f"$\Delta R$ between jets {i} and {j}",
            description=f"$\Delta R$ between jets {i} and {j}, indexed from 0",
        )

    for i in range(0, 5):
        mask = ak.num(gj, axis=1) > i
        masked_jets = gj[mask]
        masked_w = w[mask]
        htratio = masked_jets[:, i].pt / events.HT[mask]
        ret[f"pt_ht_ratio_{i}"] = makeHistogram(
            hist.axis.Regular(
                50, 0, 5, name="pt_o_ht", label=r"$\frac{p_{T}}{\text{HT}}$"
            ),
            dataset,
            htratio,
            masked_w,
            name=f"Ratio of jet {i} $p_T$ to event HT",
            description=f"Ratio of jet {i} $p_T$ to event HT",
        )
    co = lambda x: itertools.combinations(x, 2)
    for p1, p2 in co(co(range(0, 4))):
        mask = masks[p1] & masks[p2]
        p1_vals = gj[mask][:, p1[0]].phi - gj[mask][:, p1[1]].phi
        p2_vals = gj[mask][:, p2[0]].phi - gj[mask][:, p2[1]].phi
        ret["d_phi_{}{}_vs_{}{}".format(*p1, *p2)] = makeHistogram(
            [
                hist.axis.Regular(
                    50, 0, 5, name="dp1", label=r"$\Delta \phi_{" + f"{p1}" + r"}$"
                ),
                hist.axis.Regular(
                    50, 0, 5, name="dp2", label=r"$\Delta \phi_{" + f"{p2}" + r"}$"
                ),
            ],
            dataset,
            [p1_vals, p2_vals],
            w[mask],
            name=f"$\Delta \phi_{p1}$ vs $\Delta \phi_{p2}$",
        )
    return ret


def createTagHistograms(events):
    ret = {}
    dataset = events.metadata["dataset"]
    gj = events.good_jets
    w = events.EventWeight
    for name, wp in itertools.product(("tops", "bs", "Ws"), ("loose", "med", "tight")):
        ret[f"{name}_{wp}"] = makeHistogram(
            tencountaxis,
            dataset,
            ak.num(events[f"{wp}_{name}"], axis=1),
            w,
            name=f"Number of {wp} {name}",
        )
    for name, wp in itertools.product(("deep_top", "deep_W"), range(1, 5)):
        ret[f"{name}_{wp}"] = makeHistogram(
            tencountaxis,
            dataset,
            ak.num(events[f"{name}_wp{wp}"], axis=1),
            w,
            name=f"Number of wp{wp} {name}",
        )


def makePreSelectionHistograms(events):
    if "LHE" not in events.fields:
        return {}
    ret = {}
    dataset = events.metadata["dataset"]
    w = events.EventWeight
    ret[f"LHEHT"] = makeHistogram(
        ht_axis,
        dataset,
        events.LHE.HT,
        w,
        name="Event LHE HT Preselection",
        description="HT of the LHE level event before any selections are applied",
    )
    return ret


def addEventLevelVars(events):
    ht = ak.sum(events.Jet.pt, axis=1)
    events["HT"] = ht
    return events


def createEventLevelHistograms(events):
    if "LHE" not in events.fields:
        return {}
    ret = {}
    dataset = events.metadata["dataset"]
    w = events.EventWeight
    ret[f"HT"] = makeHistogram(
        ht_axis,
        dataset,
        events.HT,
        w,
        name="Event HT",
        description="Sum of $p_T$ of good AK4 jets.",
    )
    ret[f"nQLHE"] = makeHistogram(
        tencountaxis,
        dataset,
        events.LHE.Nuds + events.LHE.Nc + events.LHE.Nb,
        w,
        name="Quark Count LHE",
        description="Number of LHE level Quarks",
    )
    ret[f"nJLHE"] = makeHistogram(
        tencountaxis,
        dataset,
        events.LHE.Njets,
        w,
        name="Jet Count LHE",
        description="Number of LHE level Jets",
    )
    ret[f"nGLHE"] = makeHistogram(
        tencountaxis,
        dataset,
        events.LHE.Nglu,
        w,
        name="Gluon Count LHE",
        description="Number of LHE level gluons",
    )
    return ret


def createBHistograms(events):
    ret = {}
    dataset = events.metadata["dataset"]
    l_bjets = events.loose_bs
    m_bjets = events.med_bs

    w = events.EventWeight
    t = ak.flatten(ak.ones_like(l_bjets.pt) * w)
    ret[f"loose_bjet_pt"] = makeHistogram(
        pt_axis, dataset, ak.flatten(l_bjets.pt), t, name="Loose BJet $p_{T}$"
    )
    ret[f"loose_nb"] = makeHistogram(
        b_axis, dataset, ak.num(l_bjets.pt), w, name="Loose BJet Count"
    )
    ret[f"loose_bdr"] = makeHistogram(
        b_axis,
        dataset,
        l_bjets[:, 0].delta_r(l_bjets[:, 1]),
        w,
        name="Loose BJet $\Delta R$",
        description="$\Delta R$ between the top 2 $p_T$ b jets",
    )
    for i in range(0, 4):
        mask = ak.num(l_bjets, axis=1) > i
        ret[f"loose_b_{i}_pt"] = makeHistogram(
            pt_axis,
            dataset,
            l_bjets[mask][:,i].pt,
            w[mask],
            name=f"Loose BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(l_bjets, axis=1) > 1
    top2 = l_bjets[mask]
    lb_eta = top2[:, 0].eta - top2[:, 1].eta
    lb_phi = top2[:, 0].phi - top2[:, 1].phi
    lb_dr = top2[:, 0].delta_r(top2[:, 1])

    ret[f"loose_bb_eta"] = makeHistogram(
        eta_axis,
        dataset,
        lb_eta,
        w[mask],
        name=f"$\Delta \eta$ BB$",
        description=f"$\Delta \eta$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_phi"] = makeHistogram(
        phi_axis,
        dataset,
        lb_phi,
        w[mask],
        name=f"$\Delta \phi$ BB$",
        description=f"$\Delta \phi$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_deltar"] = makeHistogram(
        dr_axis,
        dataset,
        lb_dr,
        w[mask],
        name=f"$\Delta R$ BB$",
        description=f"$\Delta R$ between the two highest rank loose b jets",
    )
    return ret


class RPVProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):
        a_logger.debug(f"Starting analysis....")

        # w = getFileWeight(events.metadata["filename"],True)
        dataset = events.metadata["dataset"]
        events["EventWeight"] = events["MCScaleWeight"] * ak.where(
            events["genWeight"] > 0, 1, -1
        )

        pre_sel_hists = makePreSelectionHistograms(events)
        events, accum = createObjects(events)
        selection = createSelection(events)
        events = events[selection.all(*selection.names)]

        events = addEventLevelVars(events)

        if "RPV" in dataset:
            gg = goodGenParticles(events)

        jet_hists = createJetHistograms(events)
        b_hists = createBHistograms(events)
        event_hists = createEventLevelHistograms(events)
        tag_hists = createTagHistograms(events)

        return accumulate([pre_sel_hists, jet_hists, b_hists, event_hists, tag_hists])

    def postprocess(self, accumulator):
        pass


fbase = Path("samples")
samples = [
    "QCD2018",
    "Diboson2018",
    "WQQ2018",
    "ZQQ2018",
    "ST2018",
    "ZNuNu2018",
    "TT2018",
]
filesets = {
    sample: [
        f"root://cmsxrootd.fnal.gov//store/user/ckapsiak/SingleStop/Skims/Skim_2023_05_11/{sample}.root"
    ]
    for sample in samples
    # if 'Di' in sample
}

executor = processor.FuturesExecutor(workers=8)
runner = processor.Runner(executor=executor, schema=NanoAODSchema, chunksize=400000)
out = runner(filesets, "Events", processor_instance=RPVProcessor())
pickle.dump(out, open("output.pkl", "wb"))
