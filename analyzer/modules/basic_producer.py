from analyzer.core import analyzerModule
import awkward as ak

@analyzerModule("event_level", depends_on=["objects"])
def addEventLevelVars(events, analyzer):
    ht = ak.sum(events.good_jets.pt, axis=1)
    events["HT"] = ht
    return events


