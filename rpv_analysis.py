from pathlib import Path

from coffea.nanoevents import NanoAODSchema
from coffea.analysis_tools import PackedSelection
from coffea import processor
from coffea.processor import accumulate
from common import createSelection
import coffea as cf

import awkward as ak

import hist
import itertools

from collections import namedtuple
import collections.abc as cabc

import yaml

import pickle

from histograms import *
from objects import createObjects
from datasets import filesets
from weights import addWeights

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

def isGoodGenParticle(particle):
    return particle.hasFlags("isLastCopy", "fromHardProcess") & ~(
        particle.hasFlags("fromHardProcessBeforeFSR")
        & ((abs(particle.pdgId) == 1) | (abs(particle.pdgId) == 3))
    )


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
            name=fr"Composite Jet {i} to Jet {j} $\eta$",
            description=fr"$\eta$ of the sum of jets {i} to {j}",
        )
        ret[fr"m{i}{j}_m"] = makeHistogram(
            mass_axis,
            dataset,
            jets.mass,
            w,
            name=fr"Composite Jet {i} to Jet {j} mass",
            description=fr"Mass of the sum of jets {i} to {j}",
        )
    for i in range(0, 4):
        ret[fr"pt_{i}"] = makeHistogram(
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
            name=fr"$\phi$ of jet {i}",
            description=fr"$\phi$ of jet {i}(indexed from 0)",
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
        ret[fr"d_eta_{i}_{j}"] = makeHistogram(
            eta_axis,
            dataset,
            d_eta,
            w_mask,
            name=fr"$\Delta \eta$ between jets {i} and {j}",
            description=fr"$\Delta \eta$ between jets {i} and {j}, indexed from 0",
        )
        ret[f"d_phi_{i}_{j}"] = makeHistogram(
            phi_axis,
            dataset,
            d_phi,
            w_mask,
            name=fr"$\Delta \phi$ between jets {i} and {j}",
            description=fr"$\Delta \phi$ between jets {i} and {j}, indexed from 0",
        )
        ret[f"d_r_{i}_{j}"] = makeHistogram(
            dr_axis,
            dataset,
            d_r,
            w_mask,
            name=fr"$\Delta R$ between jets {i} and {j}",
            description=fr"$\Delta R$ between jets {i} and {j}, indexed from 0",
        )

    for i in range(0, 5):
        mask = ak.num(gj, axis=1) > i
        masked_jets = gj[mask]
        masked_w = w[mask]
        htratio = masked_jets[:, i].pt / events.HT[mask]
        ret[f"pt_ht_ratio_{i}"] = makeHistogram(
            hist.axis.Regular(
                50, 0, 5, name="pt_o_ht", label=r"$\frac{p_{T}}{HT}$"
            ),
            dataset,
            htratio,
            masked_w,
            name=fr"Ratio of jet {i} $p_T$ to event HT",
            description=fr"Ratio of jet {i} $p_T$ to event HT",
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
            name=fr"$\Delta \phi_{p1}$ vs $\Delta \phi_{p2}$",
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
        name=fr"Loose BJet $\Delta R$",
        description=fr"$\Delta R$ between the top 2 $p_T$ b jets",
    )
    for i in range(0, 4):
        mask = ak.num(l_bjets, axis=1) > i
        ret[f"loose_b_{i}_pt"] = makeHistogram(
            pt_axis,
            dataset,
            l_bjets[mask][:, i].pt,
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
        name=fr"$\Delta \eta$ BB$",
        description=fr"$\Delta \eta$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_phi"] = makeHistogram(
        phi_axis,
        dataset,
        lb_phi,
        w[mask],
        name=fr"$\Delta \phi$ BB$",
        description=fr"$\Delta \phi$ between the two highest rank loose b jets",
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


        events = addWeights(events)
        pre_sel_hists = makePreSelectionHistograms(events)
        events, accum = createObjects(events)
        selection = createSelection(events)
        events = events[selection.all(*selection.names)]

        events = addEventLevelVars(events)

        jet_hists = createJetHistograms(events)
        b_hists = createBHistograms(events)
        event_hists = createEventLevelHistograms(events)
        tag_hists = createTagHistograms(events)

        return accumulate([pre_sel_hists, jet_hists, b_hists, event_hists, tag_hists])

    def postprocess(self, accumulator):
        pass




if __name__ == "__main__":
    executor = processor.FuturesExecutor(workers=8)
    runner = processor.Runner(executor=executor, schema=NanoAODSchema, chunksize=400000)
    out = runner(filesets, "Events", processor_instance=RPVProcessor())
    pickle.dump(out, open("output.pkl", "wb"))
