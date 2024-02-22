from typing import Any, Dict, Optional, Tuple, Union

import dataclass
import hist
import hist.intervals as hinter
import numpy as np


@dataclass
class BasicPlottableAxis:
    edges: np.typing.NDArray[Any]
    title: Optional[str] = None
    unit: Optional[str]= None

class HistPlotObject:
    hist: hist.Hist
    title: Optional[str] = None
    style: Optional[Dict[str, Any]] = None

    def ishist(self):
        return isinstance(self.hist, hist.Hist)


    def getBinCenters(self):
        if self.ishist():
            return tuple(x.centers for x in self.hist.axes)
        else:
            edges = self.hist[1]
            ret = edges[:-1] + np.diff(edges) / 2
            ret = np.atleast_2d(ret)
            return ret

    def getBinEdges(self):
        if self.ishist():
            return tuple(x.edges for x in self.hist.axes)
        else:
            edges = self.hist[1]
            edges = np.atleast_2d(edges)
            return edges

    def getValues(self):
        if self.ishist():
            return self.hist.values()
        else:
            return self.hist[0]

    def getUncertainty(self):
        if self.ishist():
            return np.sqrt(self.hist.variances())
        else:
            return self.hist[2]

    def getStyle(self):
        return self.style.toDict() if self.style is not None else {}
