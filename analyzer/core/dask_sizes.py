from analyzer.core.results import BaseResult, SubSectorResult
from analyzer.core.histograms import HistogramCollection
import hist
from dask.sizeof import sizeof


@sizeof.register(hist.hist.Hist)
def approxSizeofHist(obj):
    return sizeof(obj.view(flow=True))


@sizeof.register(HistogramCollection)
def approxSizeofHistogramCollection(obj):
    return sizeof(obj.histogram)


@sizeof.register(BaseResult)
def approxSizeofBaseResult(obj):
    return sum(sizeof(x) for x in obj.histograms.values())


@sizeof.register(SubSectorResult)
def approxSizeofSubsectorResult(obj):
    return 1000 + sizeof(obj.base_result)
