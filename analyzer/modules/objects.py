import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.btag_points import getBTagWP
from .utils.logging import getSectorLogger


def makeCutSet(x, s, args):
    return [x[s > a] for a in args]


@MODULE_REPO.register(ModuleType.Producer)
def jets_and_ht(columns, params):

    jets = columns.get("Jet")
    gj = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]

    if "2016" in params.dataset.era.name:
        gj = gj[gj.jetId == 7]
    else:
        gj = gj[gj.jetId == 6]

    if any(x in params.dataset.era.name for x in ["2016", "2017", "2018"]):
        gj = gj[(gj.pt > 50) | ((gj.puId & 0b10) != 0)]

    good_jets = gj
    ht = ak.sum(good_jets.pt, axis=1)

    columns.add("good_jets", good_jets, shape_dependent=True)
    columns.add("HT", ht)


@MODULE_REPO.register(ModuleType.Producer)
def core_objects(columns, params):
    logger = getSectorLogger(params)
    # If all the columns are already present, ie a skim, don't bother running the module anything

    fat_jets = columns.get("FatJet")

    good_fatjets = fat_jets[(fat_jets.pt > 150) & (abs(fat_jets.eta) < 2.4)]

    bwps = getBTagWP(params)
    logger.info(f"B-tagging workign points are:\n {bwps}")
    good_jets = columns.good_jets

    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
        [bwps["L"], bwps["M"], bwps["T"]],
    )

    el = columns.get("Electron")
    mu = columns.get("Muon")
    good_electrons = el[(el.cutBased == 1) & (el.pt > 10) & (abs(el.eta) < 2.4)]
    good_muons = mu[(mu.looseId) & (mu.pfIsoId == 2) & (abs(mu.eta) < 2.4)]

    columns.add("good_fatjets", good_fatjets)
    columns.add("good_electrons", good_electrons)
    columns.add("good_muons", good_muons)
    columns.add("loose_bs", loose_b)
    columns.add("med_bs", med_b)
    columns.add("tight_bs", tight_b)


@MODULE_REPO.register(ModuleType.Producer)
def semilep_objects(columns, params):
    logger = getSectorLogger(params)
    # If all the columns are already present, ie a skim, don't bother running the module anything

    fat_jets = columns.get("FatJet")

    good_fatjets = fat_jets[(fat_jets.pt > 150) & (abs(fat_jets.eta) < 2.4)]

    bwps = getBTagWP(params)
    logger.info(f"B-tagging workign points are:\n {bwps}")
    good_jets = columns.good_jets

    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
        [bwps["L"], bwps["M"], bwps["T"]],
    )

    el = columns.get("Electron")
    mu = columns.get("Muon")
    good_electrons = el[(el.cutBased >= 3) & (el.pt > 30) & (abs(el.eta) < 2.4)]
    good_muons = mu[
        (mu.pt > 30) & (mu.mediumId) & (mu.pfIsoId == 3) & (abs(mu.eta) < 2.4)
    ]

    columns.add("good_fatjets", good_fatjets)
    columns.add("good_electrons", good_electrons)
    columns.add("good_muons", good_muons)
    columns.add("loose_bs", loose_b)
    columns.add("med_bs", med_b)
    columns.add("tight_bs", tight_b)

@MODULE_REPO.register(ModuleType.Producer)
def dijet_objects(columns, params):

    jets = columns.get("Jet")
    el = columns.get("Electron")
    mu = columns.get("Muon")
    fatjets = columns.get("FatJet")

    good_jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.5)]
    good_electrons = el[(el.pt > 50) & (el.cutBased >= 2)]
    good_muons = mu[(mu.pt > 50) & (mu.highPtId == 2) & (mu.tkRelIso < 0.1)]

    loose_b, med_b, tight_b = makeCutSet(
        good_jets,
        good_jets.btagDeepFlavB,
        [getBTagWP(params)["L"], getBTagWP(params)["M"], getBTagWP(params)["T"]],
    )

    filled_jets = ak.pad_none(good_jets, 2, axis=1)
    delta_r1 = filled_jets[:,0].delta_r(filled_jets[:,2:])
    delta_r2 = filled_jets[:,1].delta_r(filled_jets[:,2:])

    padded_r1 = ak.pad_none(delta_r1, 2, axis=1)
    padded_r2 = ak.pad_none(delta_r2, 2, axis=1)

    wide_jet1_mask = ak.fill_none((padded_r1 < 1.1) & (padded_r1 <= padded_r2), False)
    wide_jet2_mask = ak.fill_none((padded_r2 < 1.1) & (padded_r2 < padded_r1), False)

    summed_jets1 = filled_jets[wide_jet1_mask][:, 2:].sum()
    summed_jets2 = filled_jets[wide_jet2_mask][:, 2:].sum()

    wide_jet0 = filled_jets[:, 0] + summed_jets1
    wide_jet1 = filled_jets[:, 1] + summed_jets2

    columns.add('loose_bs', loose_b)
    columns.add('medium_bs', med_b)
    columns.add('tight_bs', tight_b)
    columns.add('good_electrons', good_electrons)
    columns.add('good_muons', good_muons)
    columns.add('good_jets', good_jets)
    columns.add('fat_jets', fatjets)
    columns.add('wide_jet0', wide_jet0)
    columns.add('wide_jet1', wide_jet1)