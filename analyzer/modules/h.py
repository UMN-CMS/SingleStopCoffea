import itertools as it

import awkward as ak
import hist
from analyzer.core import MODULE_REPO, ModuleType
import vector
import numpy as np

from .utils.axes import makeAxis

@MODULE_REPO.register(ModuleType.Producer)
def dijet_deltar(columns, params):
    good_jets = columns.get("good_jets")
    mask = ak.num(good_jets) >= 4 & ak.num(good_jets) <= 8
    four_good_jets = ak.where(mask, good_jets, good_jets[:0])    
    jet_indices = ak.local_index(four_good_jets.pt)
    dijet_combos = ak.combinations(four_good_jets, 2)
    dijet_idx_combos = ak.combinations(jet_indices, 2)
    dijet_delta_r = dijet_combos['0'].delta_r(dijet_combos['1'])
    dijets = ak.zip({
        "vec": dijet_combos["0"] + dijet_combos["1"],
        "jets": dijet_combos,
        "idx": dijet_idx_combos,
        "delta_r": dijet_delta_r,
    })
    dijet_pair_combos = ak.combinations(dijets, 2)
    def no_shared_jets(idx1, idx2):
        return (
            (idx1["0"] != idx2["0"]) &
            (idx1["0"] != idx2["1"]) &
            (idx1["1"] != idx2["0"]) &
            (idx1["1"] != idx2["1"])
        )
    mask = no_shared_jets(dijet_pair_combos["0"].idx, dijet_pair_combos["1"].idx)
    dijet_pair_combos = dijet_pair_combos[mask]
    dijet_pair_delta_r = dijet_pair_combos["0"].vec.delta_r(dijet_pair_combos["1"].vec)
    dijet_pairs = ak.zip({
        "vec": dijet_pair_combos["0"].vec + dijet_pair_combos["1"].vec,
        "delta_r": dijet_pair_delta_r,
    })
    columns.add('dijets', dijets, shape_dependent=True)
    columns.add('dijet_pairs', dijet_pairs, shape_dependent=True)


@MODULE_REPO.register(ModuleType.Histogram)
def dijet_plots(events, params, analyzer):
    analyzer.H(
        f"dijet_pt",
        makeAxis(200, 0, 2000, "Dijet $p_{T}$", unit="GeV"),
        events.dijets.vec.pt,
        description="pt of dijets",
    )
    analyzer.H(
        f"dijet_delta_r",
        makeAxis(50, 0, 6, r"Dijet $\Delta R$"),
        events.dijets.delta_r,
        description="delta r of dijets",
    )
    analyzer.H(
        f"dijet_pair_pt",
        makeAxis(400, 0, 4000, "Dijet Pair $p_{T}$", unit="GeV"),
        events.dijet_pairs.vec.pt,
        description="pt of dijet pairs"
    )
    analyzer.H(
        f"dijet_pair_delta_r",
        makeAxis(50, 0, 6, r"Dijet Pair $\Delta R$"),
        events.dijet_pairs.delta_r,
        description="delta r of dijet pairs",
    )

@MODULE_REPO.register(ModuleType.Histogram)
def jet_plots(events, params, analyzer):
    jets = events.good_jets
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
        f"jet_ht",
        makeAxis(50, 0, 2000, r"Jet $H_T$", unit="GeV"),
        events.HT,
        description="ht of all good jets",
    )

@MODULE_REPO.register(ModuleType.Selection)
def signal_selection(events, params, selector):
    good_jets = events.good_jets
    selector.add(f"≥4 good jets", ak.num(good_jets) >= 4)

@MODULE_REPO.register(ModuleType.Selection)
def signal_preselection(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    selector.add(
        f"every event", events.HLT[ht_trigger_name] | ~events.HLT[ht_trigger_name]
    )
@MODULE_REPO.register(ModuleType.Selection)
def trivial_selection(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    selector.add(
        f"every event also", events.HLT[ht_trigger_name] | ~events.HLT[ht_trigger_name]
    )