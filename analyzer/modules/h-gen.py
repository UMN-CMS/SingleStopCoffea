import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.axes import makeAxis


@MODULE_REPO.register(ModuleType.Histogram)
def all_fatjets(events, params, analyzer):
    fatjets = events.good_fatjets
    analyzer.H(
        f"fatjet_pt",
        makeAxis(50, 140, 1000, "AK8 Jet $p_{T}$", unit="GeV"),
        fatjets.pt,
        description="pt of fatjets",
    )
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
def jet_multiplicity(events, params, analyzer):
    njets = ak.num(events.good_jets, axis=1)
    analyzer.H(
        "njets",
        makeAxis(16, -0.5, 15.5, "AK4 $N_{jets}$"),
        njets,
        description="AK4 jet multiplicity"
    )

@MODULE_REPO.register(ModuleType.Histogram)
def gen_higgs(events, params, analyzer):
    genpart = events.GenPart
    mask = (genpart.pdgId == 25) & (((genpart.statusFlags>>13)&1)==1)
    genhiggs = genpart[mask]
    analyzer.H(
        f"genhiggs_pt",
        makeAxis(50, 0, 500, "Truth Higgs $p_{T}$", unit="GeV"),
        genhiggs.pt,
        description=f"pt of truth level higgs bosons"
    )

