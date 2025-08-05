import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType

from .utils.btag_points import getBTagWP
from .utils.logging import getSectorLogger


def makeCutSet(x, s, args):
    return [x[s > a] for a in args]


@MODULE_REPO.register(ModuleType.Producer)
def jets_and_ht(columns, params):

    jets = columns.get("Jet")
    good_jets = jets[(jets.pt > 30) & (abs(jets.eta) < 2.4)]
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
def dijet_objects(columns, params):

    jets = columns.get("Jet")
    el = columns.get("Electron")
    mu = columns.get("Muon")
    #fatjets = columns.get("FatJet")

    good_jets_w_lep = jets[(jets.pt > 30) & (abs(jets.eta) < 2.5)]
    iso_electrons = el[(el.pt > 50) & (el.cutBased >= 2)]
    iso_muons = mu[(mu.pt > 50) & (mu.highPtId == 2) & (mu.tkRelIso < 0.1)]

    near_el = good_jets_w_lep.nearest(iso_electrons,threshold=0.4)
    near_mu = good_jets_w_lep.nearest(iso_muons,threshold=0.4)
    good_jets=good_jets_w_lep[ak.is_none(near_el,axis=-1) & ak.is_none(near_mu,axis=-1)]

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

    #columns.add('loose_bs', loose_b)
    #columns.add('tight_bs', tight_b)
    #columns.add('fat_jets', fatjets)
    columns.add('good_jets', good_jets)
    columns.add('medium_bs', med_b)
    columns.add('wide_jet0', wide_jet0)
    columns.add('wide_jet1', wide_jet1)