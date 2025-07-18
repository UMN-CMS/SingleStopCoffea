import itertools as it

import awkward as ak
import hist
from analyzer.core import MODULE_REPO, ModuleType

from .utils.axes import makeAxis

@MODULE_REPO.register(ModuleType.Producer)
def electrons_muons(columns, params):
    electrons = columns.get("Electron")
    good_electrons = electrons[electrons.pt > 12]
    dielec = ak.combinations(electrons, 2, fields=["e1", "e2"])
    dielec_mass = (dielec["e1"] + dielec["e2"]).mass

    muons = columns.get("Muon")
    good_muons = muons[muons.pt > 12]
    dimuon = ak.combinations(muons, 2, fields=["e1", "e2"])
    dimuon_mass = (dimuon["e1"] + dimuon["e2"]).mass
    
    columns.add("good_electrons", good_electrons, shape_dependent=True)
    columns.add("dielec_mass", dielec_mass, shape_dependent=True)
    columns.add("good_muons", good_muons, shape_dependent=True)
    columns.add("dimuon_mass", dimuon_mass, shape_dependent=True)

@MODULE_REPO.register(ModuleType.Histogram)
def pt_plots(events, params, analyzer):
    analyzer.H(
        f"electron_pt",
        makeAxis(40, 0, 250, "$p_{T}$", unit="GeV"),
        events.good_electrons.pt,
        description="pt of electrons",
    )
    analyzer.H(
        f"muon_pt",
        makeAxis(40, 0, 250, "$p_{T}$", unit="GeV"),
        events.good_muons.pt,
        description="pt of muons",
    )
    
@MODULE_REPO.register(ModuleType.Histogram)
def dimass_plots(events, params, analyzer):
    analyzer.H(
        f"dielec_mass",
        makeAxis(40, 0, 400, r"$m_{ee}$", unit="GeV"),
        events.dielec_mass,
        description= "mass of di-electrons",
    )
    analyzer.H(
        f"dimuon_mass",
        makeAxis(40, 0, 400, r"$m_{\mu\mu}$", unit="GeV"),
        events.dimuon_mass,
        description="mass of dimuons",
    )

@MODULE_REPO.register(ModuleType.Selection)
def trivial_selection(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    selector.add(
        f"every event", events.HLT[ht_trigger_name] | ~events.HLT[ht_trigger_name]
    )

@MODULE_REPO.register(ModuleType.Selection)
def trivial_selection2(events, params, selector):
    era_info = params.dataset.era
    ht_trigger_name = era_info.trigger_names["HT"]
    selector.add(
        f"every event2", events.HLT[ht_trigger_name] | ~events.HLT[ht_trigger_name]
    )