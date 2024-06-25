import awkward as ak

from analyzer.core import analyzerModule


@analyzerModule("event_level", depends_on=["objects"])
def addEventLevelVars(events, analyzer):
    ht = ak.sum(events.good_jets.pt, axis=1)
    events["HT"] = ht
    return events, analyzer


