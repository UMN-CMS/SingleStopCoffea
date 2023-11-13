from analyzer.core import analyzerModule, ModuleType
import awkward as ak

@analyzerModule("event_level", ModuleType.MainProducer)
def addEventLevelVars(events):
    ht = ak.sum(events.good_jets.pt, axis=1)
    events["HT"] = ht
    return events


