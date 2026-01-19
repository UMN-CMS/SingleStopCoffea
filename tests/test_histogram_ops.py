import pytest
import hist
import numpy as np
from analyzer.cli.browser import processHistogram


def testProcessHistogramSum():
    h = hist.Hist.new.Regular(10, 0, 10, name="x").Regular(10, 0, 10, name="y").Double()
    h.fill(x=np.random.uniform(0, 10, 100), y=np.random.uniform(0, 10, 100))

    # Sum over Y (axis 1), keep X (axis 0)
    config = {0: {"action": "keep"}, 1: {"action": "sum"}}

    res = processHistogram(h, config)
    assert len(res.axes) == 1
    assert res.axes[0].name == "x"
    assert res.sum() == h.sum()


def testProcessHistogramSlice():
    h = hist.Hist.new.Regular(10, 0, 10, name="x").Regular(10, 0, 10, name="y").Double()
    # Fill specific bin
    # Bin 0 for x is [0, 1)
    # Bin 0 for y is [0, 1)
    h.fill(x=[0.5], y=[0.5])  # Bin (0,0) -> 1
    h.fill(x=[0.5], y=[1.5])  # Bin (0,1) -> 1

    # Slice Y at bin 0. Keep X.
    config = {0: {"action": "keep"}, 1: {"action": "slice", "bin": 0}}

    res = processHistogram(h, config)
    assert len(res.axes) == 1
    assert res.axes[0].name == "x"
    # Result should correspond to Y=0 slice.
    # At X=0 (bin 0), we have 1 count from (0.5, 0.5).
    # (0.5, 1.5) is in Y bin 1, so should not be in this slice.

    assert res[0] == 1.0
    assert res.sum() == 1.0


def testProcessHistogramDefaults():
    # If config missing, defaults to keep
    h = hist.Hist.new.Regular(5, 0, 5, name="x").Double()
    res = processHistogram(h, {})
    assert len(res.axes) == 1
