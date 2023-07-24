from analyzer.core import analyzerModule, ModuleType
from analyzer.matching import object_matching
import awkward as ak

@analyzerModule("event_level", ModuleType.MainProducer)
def addEventLevelVars(events):
    ht = ak.sum(events.Jet.pt, axis=1)
    events["HT"] = ht
    return events


@analyzerModule("delta_r", ModuleType.MainProducer,require_tags=["signal"], after=["good_gen"])
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
