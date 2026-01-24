import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.axes import makeAxis

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

