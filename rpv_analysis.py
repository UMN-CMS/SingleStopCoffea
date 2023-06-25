from pathlib import Path

import sys
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
from objects import createObjects, b_tag_wps
from datasets import filesets
from weights import addWeights
from matching import object_matching

import itertools as it

import warnings

warnings.filterwarnings("ignore", message=r".*Missing cross-reference")

ParticleChain = namedtuple("ParticleChain", "name pdgId children")
ParticleMatch = namedtuple("ParticleMatch", "name pdgId matched_array")

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

# order stop b , chi b, chi s, chi d


def isGoodGenParticle(particle):
    return particle.hasFlags("isLastCopy", "fromHardProcess") & ~(
        particle.hasFlags("fromHardProcessBeforeFSR")
        & ((abs(particle.pdgId) == 1) | (abs(particle.pdgId) == 3))
    )


def createGoodChildren(gen_particles, children):
    # x = ak.singletons(gen_particles.children)
    good_child_mask = isGoodGenParticle(children)
    while not ak.all(ak.ravel(good_child_mask)):
        good_child_mask = isGoodGenParticle(children)
        maybe_good_children = children[~good_child_mask].children
        maybe_good_children = ak.flatten(maybe_good_children, axis=3)
        children = ak.concatenate(
            [children[good_child_mask], maybe_good_children], axis=2
        )
    return children


def genMatchParticles(children_particles, children_structure, allow_anti=True):
    # base_particle_array = base_particle.matched_array
    # genpart_children = base_particle_array.children
    ret = {}
    groups = it.groupby(
        children_structure, key=lambda x: abs(x.pdgId) if allow_anti else x.pdgId
    )
    for pid, children in groups:
        children = list(children)
        mask = pid == (
            abs(children_particles.pdgId) if allow_anti else children_particles.pdgId
        )
        if ak.any(ak.count(mask[mask], axis=1) != len(children)):
            raise Exception(
                f"Did not find expected number of children of type {pid} in event. Expected {len(children)}."
            )
        matched = children_particles[mask]
        for i, c in enumerate([c for c in children]):
            ret[c.name] = matched[:, i]
            if c.children:
                print(
                    f"Looking for children of particle {c.pdgId} -> {[x.pdgId for x in c.children]} in list : {[x.pdgId for x in matched[:,i].good_children[0]]}"
                )
                print(
                    f"Looking for children of particle {c.pdgId} -> {[x.pdgId for x in c.children]} in list : {[x.pdgId for x in matched[:,i].good_children[0]]}"
                )
                ret.update(
                    genMatchParticles(
                        matched[:, i].good_children, c.children, allow_anti=allow_anti
                    )
                )
    print(f"Ret is {ret}")
    return ret

    # ret = {}
    # for s_child in structure:
    #    found = children[
    #        s_child.pdgid == abs(children.pdgId)  if allow_anti else children.pdgId
    #        & children.parent
    #    ]
    #    print(found)
    #    if ak.any(ak.num(found, axis=1) != 1):
    #        raise ValueError(f"Could not find particle with pdgId {s_child.pdgid}")
    #    ret[s_child.name] = found
    #    if s_child.children:
    #        ret.update(genMatchParticles(found, s_child.children))
    # return ret


def deltaRMatch(events):
    # ret =  object_matching(events.SignalQuarks, events.good_jets, 0.3, None, False)
    matched_jets, matched_quarks, dr, idx_j, idx_q, _ = object_matching(
        events.good_jets, events.SignalQuarks, 0.3, 0.5, True
    )
    events["matched_quarks"] = matched_quarks
    events["matched_jets"] = matched_jets
    events["matched_dr"] = dr
    events["matched_jet_idx"] = idx_j
    return events


def goodGenParticles(events):
    test = createGoodChildren(events.GenPart, events.GenPart.children)

    def get(x):
        return [y.pdgId for y in x[0]]

    events["GenPart", "good_children"] = test
    gg = events.GenPart[isGoodGenParticle(events.GenPart)]
    bs = gg.good_children[abs(gg.good_children.pdgId) == 5]
    stop = ak.all(abs(gg[:, 0].pdgId) == 1000006)
    bad = ak.all(abs(gg[:, 1].pdgId) == 1000024)
    x = gg[abs(gg.pdgId) == 1000024][:, 0]
    t = gg[abs(gg.pdgId) == 1000006][:, 0]
    x_children = gg.good_children[abs(gg.pdgId) == 1000024]
    s_children = gg.good_children[abs(gg.pdgId) == 1000006]
    xb = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 5]))
    xd = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 1]))
    xs = ak.flatten(ak.flatten(x_children[abs(x_children.pdgId) == 3]))
    sb = ak.flatten(ak.flatten(s_children[abs(s_children.pdgId) == 5]))
    events["SignalParticles"] = ak.zip(
        dict(
            stop=t,
            chi=x,
            stop_b=sb,
            chi_b=xb,
            chi_d=xd,
            chi_s=xs,
        )
    )
    events["SignalQuarks"] = ak.concatenate(
        [ak.singletons(val) for val in [sb, xb, xd, xs]], axis=1
    )
    return events


def createSignalHistograms(events, hmaker):
    ret = {}
    chi_match_axis = hist.axis.IntCategory(
        [0, 1, 2, 3], name="num_matched_chi", label=r"Number of reco chi jets correct"
    )
    num_top_3_are_chi = ak.num(
        events.matched_jet_idx[:, 1:][events.matched_jet_idx[:, 1:] < 3], axis=1
    )
    num_sub_3_are_chi = ak.num(
        events.matched_jet_idx[:, 1:][
            (events.matched_jet_idx[:, 1:] < 4) & (events.matched_jet_idx[:, 1:] > 0)
        ],
        axis=1,
    )
    ret[f"num_top_3_jets_matched_chi_children"] = hmaker(
        chi_match_axis, num_top_3_are_chi, name="num_top_3_are_chi"
    )
    ret[f"num_sub_3_jets_matched_chi_children"] = hmaker(
        chi_match_axis, num_sub_3_are_chi, name="num_sub_3_are_chi"
    )

    idx_axis = hist.axis.IntCategory(range(0, 8), name="Jet Idx", label=r"Jet idx")
    e = events.matched_jet_idx[:, 1]
    ret[f"chi_b_jet_idx"] = hmaker(idx_axis, ak.fill_none(e, 7), name="chi_b_jet_idx")
    return ret


def charginoRecoHistograms(events, hmaker):
    ret = {}
    gj = events.good_jets
    w = events.EventWeight
    idx = ak.local_index(gj, axis=1)
    med_bjet_mask = gj.btagDeepFlavB > b_tag_wps[1]

    t_lead_b_idx = idx[med_bjet_mask]
    lead_b_idx = t_lead_b_idx[:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]
    no_lead_idxs = idx[idx != lead_b_idx]
    no_sublead_idxs = idx[idx != sublead_b_idx]
    no_lead_or_sublead_idxs = idx[(idx != sublead_b_idx) & (idx != lead_b_idx)]
    no_lead_jets = gj[no_lead_idxs]
    print(no_lead_idxs)
    print(lead_b_idx)
    print(gj[ak.singletons(lead_b_idx)][:,0])
    no_sublead_jets = gj[no_sublead_idxs]
    first, second = ak.unzip(ak.combinations(no_lead_jets[:, 0:3], 2))
    max_dr_no_lead = ak.max(first.delta_r(second), axis=1)
    first, second = ak.unzip(ak.combinations(no_sublead_jets[:, 0:3], 2))
    max_dr_no_sublead = ak.max(first.delta_r(second), axis=1)
    max_no_lead_over_max_sublead = max_dr_no_lead / max_dr_no_sublead

    ret[f"m3_top_3_no_lead_b"] = hmaker(
        mass_axis,
        (no_lead_jets[:, 0:3].sum()).mass,
        name="m3_top_3_no_lead_b",
    )

    ret[f"m3_top_3_no_b_unless_dR_charg_gt_2"] = hmaker(
        mass_axis,
        ak.where(max_no_lead_over_max_sublead > 2,
                (no_lead_jets[:, 0:3].sum()).mass,
                (gj[no_lead_or_sublead_idxs][:, 0:2].sum() + gj[ak.singletons(lead_b_idx)][:,0]).mass
            
        ),
        name="m3_top3_no_b_unless_dR_charg_gt_2",
    )

    ret[f"m3_top_2_plus_lead_b"] = hmaker(
        mass_axis,
        (no_lead_jets[:, 0:2].sum() +gj[ak.singletons(lead_b_idx)][:,0]).mass,
        name="m3_top_2_plus_lead_b",
    )
    return ret


def createJetHistograms(events, hmaker):
    ret = {}
    dataset = events.metadata["dataset"]
    gj = events.good_jets
    w = events.EventWeight

    ret[f"h_njet"] = hmaker(nj_axis, ak.num(gj), name="njets")
    jet_combos = [(0, 4), (0, 3), (1, 4)]
    co = lambda x: itertools.combinations(x, 2)

    masses = {}

    for i, j in jet_combos:
        jets = gj[:, i:j].sum()
        masses[(i, j)] = jets.mass
        ret[f"m{i+1}{j}_pt"] = hmaker(
            pt_axis,
            jets.pt,
            name=f"Composite Jet {i+1} to Jet {j} $p_T$",
            description=f"$p_T$ of the sum of jets {i+1} to {j}",
        )
        ret[f"m{i+1}{j}_eta"] = hmaker(
            eta_axis,
            jets.eta,
            name=rf"Composite Jet {i+1} to Jet {j} $\eta$",
            description=rf"$\eta$ of the sum of jets {i+1} to {j}",
        )
        ret[rf"m{i+1}{j}_m"] = hmaker(
            mass_axis,
            jets.mass,
            name=rf"Composite Jet {i+1} to Jet {j} mass",
            description=rf"Mass of the sum of jets {i+1} to {j}",
        )

    for p1, p2 in co(jet_combos):
        p1_1, p1_2 = p1
        p2_1, p2_2 = p2
        m1 = hist.axis.Regular(
            150,
            0,
            3000,
            name=f"mass_{p1_1+1}{p1_2}",
            label=rf"$m_{{{p1_1 + 1}{p1_2}}}$ [GeV]",
        )
        m2 = hist.axis.Regular(
            150,
            0,
            3000,
            name=f"mass_{p2_1+1}{p2_2}",
            label=rf"$m_{{{p2_1 + 1}{p2_2}}}$ [GeV]",
        )
        ratio = hist.axis.Regular(
            50,
            0,
            1,
            name=f"ratio",
            label=rf"$\frac{{m_{{ \sum {p2_1+1}{p2_2} }} }}{{ m_{{ \sum {p1_1+1}{p1_2} }} }}$ [GeV]",
        )
        ret[f"m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}"] = hmaker(
            [m1, m2], [masses[p1], masses[p2]], name="Comp mass"
        )

        ret[f"ratio_comp_m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}"] = hmaker(
            [m1, ratio], [masses[p1], masses[p2] / masses[p1]], name="mass2dratio"
        )

    for i in range(0, 4):
        ret[rf"pt_{i}"] = hmaker(
            pt_axis,
            gj[:, i].pt,
            name=f"$p_T$ of jet {i+1}",
            description=f"$p_T$ of jet {i+1} ",
        )
        ret[f"eta_{i}"] = hmaker(
            eta_axis,
            gj[:, i].eta,
            name=f"$\eta$ of jet {i+1}",
            description=f"$\eta$ of jet {i+1}",
        )
        ret[f"phi_{i}"] = hmaker(
            phi_axis,
            gj[:, i].phi,
            name=rf"$\phi$ of jet {i+1}",
            description=rf"$\phi$ of jet {i+1}",
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


def createTagHistograms(events, hmaker):
    ret = {}
    dataset = events.metadata["dataset"]
    gj = events.good_jets
    w = events.EventWeight
    for name, wp in itertools.product(("tops", "bs", "Ws"), ("loose", "med", "tight")):
        ret[f"{name}_{wp}"] = hmaker(
            tencountaxis,
            ak.num(events[f"{wp}_{name}"], axis=1),
            name=f"Number of {wp} {name}",
        )
    for name, wp in itertools.product(("deep_top", "deep_W"), range(1, 5)):
        ret[f"{name}_{wp}"] = hmaker(
            tencountaxis,
            ak.num(events[f"{name}_wp{wp}"], axis=1),
            name=f"Number of wp{wp} {name}",
        )


def makePreSelectionHistograms(events, hmaker):
    if "LHE" not in events.fields:
        return {}
    ret = {}
    dataset = events.metadata["dataset"]
    w = events.EventWeight
    # ret[f"LHEHT"] = hmaker(
    #    ht_axis,
    #    events.LHE.HT,
    #    w,
    #    name="Event LHE HT Preselection",
    #    description="HT of the LHE level event before any selections are applied",
    # )
    return ret


def addEventLevelVars(events):
    ht = ak.sum(events.Jet.pt, axis=1)
    events["HT"] = ht
    return events


def createEventLevelHistograms(events, hmaker):
    dataset = events.metadata["dataset"]
    ret = {}
    ret[f"HT"] = hmaker(
        ht_axis,
        events.HT,
        name="Event HT",
        description="Sum of $p_T$ of good AK4 jets.",
    )
    if "LHE" not in events.fields:
        return ret
    ret[f"nQLHE"] = hmaker(
        tencountaxis,
        events.LHE.Nuds + events.LHE.Nc + events.LHE.Nb,
        name="Quark Count LHE",
        description="Number of LHE level Quarks",
    )
    ret[f"nJLHE"] = hmaker(
        tencountaxis,
        events.LHE.Njets,
        name="Jet Count LHE",
        description="Number of LHE level Jets",
    )
    ret[f"nGLHE"] = hmaker(
        tencountaxis,
        events.LHE.Nglu,
        name="Gluon Count LHE",
        description="Number of LHE level gluons",
    )
    return ret


def createBHistograms(events, hmaker):
    ret = {}
    dataset = events.metadata["dataset"]
    l_bjets = events.loose_bs
    m_bjets = events.med_bs

    ret[f"loose_bjet_pt"] = hmaker(pt_axis, l_bjets.pt, name="Loose BJet $p_{T}$")
    ret[f"loose_nb"] = hmaker(b_axis, ak.num(l_bjets.pt), name="Loose BJet Count")
    ret[f"loose_bdr"] = hmaker(
        b_axis,
        l_bjets[:, 0].delta_r(l_bjets[:, 1]),
        name=rf"Loose BJet $\Delta R$",
        description=rf"$\Delta R$ between the top 2 $p_T$ b jets",
    )
    for i in range(0, 4):
        mask = ak.num(l_bjets, axis=1) > i
        ret[f"loose_b_{i}_pt"] = hmaker(
            pt_axis,
            l_bjets[mask][:, i].pt,
            mask=mask,
            name=f"Loose BJet {i} $p_T$",
            description=f"$p_T$ of the rank {i} $p_T$ b jet",
        )
    mask = ak.num(l_bjets, axis=1) > 1
    top2 = l_bjets[mask]
    lb_eta = top2[:, 0].eta - top2[:, 1].eta
    lb_phi = top2[:, 0].phi - top2[:, 1].phi
    lb_dr = top2[:, 0].delta_r(top2[:, 1])

    ret[f"loose_bb_eta"] = hmaker(
        eta_axis,
        lb_eta,
        name=rf"$\Delta \eta$ BB$",
        description=rf"$\Delta \eta$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_phi"] = hmaker(
        phi_axis,
        lb_phi,
        name=rf"$\Delta \phi$ BB$",
        description=rf"$\Delta \phi$ between the two highest rank loose b jets",
    )
    ret[f"loose_bb_deltar"] = hmaker(
        dr_axis,
        lb_dr,
        name=f"$\Delta R$ BB$",
        description=f"$\Delta R$ between the two highest rank loose b jets",
    )
    return ret


def makeCategoryHist(cat_axes, cat_vals, event_weights):
    def internal(
        axis,
        data,
        mask=None,
        name=None,
        description=None,
        auto_expand=True,
    ):
        # print("###############################################")
        # print(f"name: {name}")
        # print(f"axes: {axis}")
        # print(f"Cat axes: {cat_axes}")
        if not isinstance(data, list):
            data = [data]
        if not data:
            raise Exception("No data")
        if isinstance(axis, list):
            all_axes = cat_axes + axis
        else:
            all_axes = cat_axes + [axis]
        h = hist.Hist(*all_axes, storage="weight", name=name)
        setattr(h, "description", description)

        weights = event_weights[mask] if mask is not None else event_weights
        base_category_vals = cat_vals
        if mask is not None:
            base_category_vals = [
                x[mask] if isinstance(x, ak.Array) else x for x in base_category_vals
            ]
        shaped_cat_vals = base_category_vals
        shaped_data_vals = data
        if auto_expand:
            mind, maxd = data[0].layout.minmax_depth
            if maxd > 1:
                ol = ak.ones_like(data[0])
                weights = ak.flatten(ol * weights)
                shaped_cat_vals = [
                    ak.flatten(ol * x) if isinstance(x, ak.Array) else x
                    for x in cat_vals
                ]
                shaped_data_vals = [
                    ak.flatten(x) if isinstance(x, ak.Array) else x
                    for x in shaped_data_vals
                ]
        d = shaped_cat_vals + shaped_data_vals
        # print(f"{name} HIST: {h}")
        # print(f"{name} DATA: {d}")
        # print(f"{name} WEIGHTS: {weights}")
        ret = h.fill(*d, weight=weights)
        return ret

    return internal


class RPVProcessor(processor.ProcessorABC):
    def __init__(self):
        pass

    def process(self, events):

        events = addWeights(events)
        pre_sel_hists = makePreSelectionHistograms(events, makeHistogram)
        events = createObjects(events)
        selection = createSelection(events)
        events = events[selection.all(*selection.names)]
        events = addEventLevelVars(events)

        dataset = events.metadata["dataset"]

        to_accumulate = []

        if "signal" in dataset and True:
            events = goodGenParticles(events)
            events = deltaRMatch(events)
            matched_cat = ak.num(
                events.matched_quarks[~ak.is_none(events.matched_quarks, axis=1)],
                axis=1,
            )
            hm = makeCategoryHist(
                [
                    dataset_axis,
                    hist.axis.IntCategory([4, 5, 6], name="number_jets", label="NJets"),
                    hist.axis.IntCategory(
                        [0, 1, 2, 3, 4], name="num_matched", label="Num Matched"
                    ),
                ],
                [dataset, ak.num(events.good_jets, axis=1), matched_cat],
                events.EventWeight,
            )
            sig_hists = createSignalHistograms(events, hm)
            charg_hists = charginoRecoHistograms(events, hm)
            to_accumulate.append(sig_hists)
            to_accumulate.append(charg_hists)
        else:
            hm = makeCategoryHist(
                [
                    dataset_axis,
                    hist.axis.IntCategory([4, 5, 6], name="number_jets", label="NJets"),
                ],
                [dataset, ak.num(events.good_jets, axis=1)],
                events.EventWeight,
            )
        jet_hists = createJetHistograms(events, hm)
        b_hists = createBHistograms(events, hm)
        event_hists = createEventLevelHistograms(events, hm)
        tag_hists = createTagHistograms(events, hm)
        to_accumulate.extend([jet_hists, b_hists, event_hists, tag_hists])
        return accumulate(to_accumulate)

    def postprocess(self, accumulator):
        pass


if __name__ == "__main__":
    executor = processor.IterativeExecutor()
    executor = processor.FuturesExecutor(workers=4)
    runner = processor.Runner(executor=executor, schema=NanoAODSchema, chunksize=400000)
    # f = {k: v for k, v in filesets.items()}
    f = {k: v for k, v in filesets.items() if "signal" in k}
    # f = {k: v for k, v in filesets.items() if "signal_2000_1900" in k}  
    out = runner(f, "Events", processor_instance=RPVProcessor())
    pickle.dump(out, open("signal_hists.pkl", "wb"))
