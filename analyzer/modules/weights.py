import awkward as ak
import dask_awkward as dak

from analyzer.core import analyzerModule


@analyzerModule("weights", categories="weights")
def addWeights(events, analyzer):
    if "genWeight" in events.fields:
        analyzer.weights.add("PosNeg", dak.where(events.genWeight > 0, 1, -1))
    return events, analyzer
