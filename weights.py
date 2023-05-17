import awkward as ak

def addWeights(events):
    dataset = events.metadata["dataset"]
    events["EventWeight"] = events["MCScaleWeight"] * ak.where(
            events["genWeight"] > 0, 1, -1
        )
    return events
