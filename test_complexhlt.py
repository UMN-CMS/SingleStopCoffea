import sys
import os
import numpy as np
import unittest.mock as mock

# Add project root to path
project_root = "/uscms/home/ckapsiak/nobackup/Analysis/SingleStop/SingleStopCoffea"
if project_root not in sys.path:
    sys.path.append(project_root)


# Mock HLT array support to be used in ak.full_like mock
class MockHLTArray:
    def __init__(self, arr):
        self.arr = np.array(arr, dtype=bool)

    def __or__(self, other):
        if isinstance(other, bool):
            return MockHLTArray(self.arr | other)
        return MockHLTArray(self.arr | other.arr)

    def __and__(self, other):
        if isinstance(other, bool):
            return MockHLTArray(self.arr & other)
        return MockHLTArray(self.arr & other.arr)

    def __invert__(self):
        return MockHLTArray(~self.arr)

    def __repr__(self):
        return str(self.arr)


# Mock awkward
# We have to mock it because it is imported at module level in hlt_selection.py
mock_ak = mock.MagicMock()


def full_like_side_effect(array, value):
    # Mimic ak.full_like behavior for our MockHLTArray
    if hasattr(array, "arr"):
        return MockHLTArray(np.full_like(array.arr, value, dtype=bool))
    # Fallback if somehow real array passed (unlikely in this mock setup)
    return MockHLTArray(np.array([value], dtype=bool))


mock_ak.full_like.side_effect = full_like_side_effect
sys.modules["awkward"] = mock_ak

try:
    from analyzer.modules.common.hlt_selection import ComplexHLT, ComplexHLTConfig
    from analyzer.utils.querying import Pattern, PatternMode
except ImportError as e:
    print(f"Import failed: {e}")
    sys.exit(1)

import functools as ft
import operator as op


# Mock HLT dict
class MockHLT:
    def __init__(self, data):
        self.data = data

    def __getitem__(self, key):
        return self.data[key]


# Mock Columns object
class MockColumns:
    def __init__(self, metadata, hlt_data):
        self.metadata = metadata
        self.data = {"HLT": hlt_data}
        self.selections = {}

    def __getitem__(self, key):
        return self.data[key]

    def addColumnsFrom(self, other, outputs):
        pass


import analyzer.modules.common.hlt_selection as target_module


def mock_addSelection(columns, name, selection):
    columns.selections[name] = selection


target_module.addSelection = mock_addSelection


def test():
    # Setup Data
    t1 = MockHLTArray([1, 0, 1, 0, 1])
    t2 = MockHLTArray([0, 1, 1, 0, 0])
    t3 = MockHLTArray([0, 0, 0, 1, 0])

    hlt_data = MockHLT({"T1_bit": t1, "T2_bit": t2, "T3_bit": t3})

    trigger_names = {"Trig1": "T1_bit", "Trig2": "T2_bit", "Trig3": "T3_bit"}

    # Case 1: Standard Glob (implicit)
    print("--- Test 1: Implicit Glob ---")
    config1 = [
        ComplexHLTConfig(
            pattern=Pattern(pattern="SetA*", mode=PatternMode.GLOB), triggers=["Trig1"]
        )
    ]
    mod1 = ComplexHLT(trigger_config=config1)
    cols1 = MockColumns(
        {"dataset_name": "SetA_Run2018", "era": {"trigger_names": trigger_names}},
        hlt_data,
    )
    mod1.run(cols1, {})
    print("Result:", cols1.selections["PassHLT"].arr)
    assert np.all(cols1.selections["PassHLT"].arr == [True, False, True, False, True])

    # Case 2: Regex
    print("--- Test 2: Regex ---")
    config2 = [
        ComplexHLTConfig(
            pattern=Pattern(pattern="Set[AB].*", mode=PatternMode.REGEX),
            triggers=["Trig2"],
        )
    ]
    mod2 = ComplexHLT(trigger_config=config2)
    cols2 = MockColumns(
        {"dataset_name": "SetB_Run2018", "era": {"trigger_names": trigger_names}},
        hlt_data,
    )
    mod2.run(cols2, {})
    print("Result:", cols2.selections["PassHLT"].arr)
    assert np.all(cols2.selections["PassHLT"].arr == [False, True, True, False, False])

    # Case 3: Veto
    print("--- Test 3: Veto ---")
    config3 = [
        ComplexHLTConfig(
            pattern=Pattern(pattern="SetB*", mode=PatternMode.GLOB),
            triggers=["Trig2"],
            veto=["Trig1"],
        )
    ]
    mod3 = ComplexHLT(trigger_config=config3)
    cols3 = MockColumns(
        {"dataset_name": "SetB_Run2018", "era": {"trigger_names": trigger_names}},
        hlt_data,
    )
    mod3.run(cols3, {})
    # Trig2: [0, 1, 1, 0, 0]
    # Trig1: [1, 0, 1, 0, 1]
    # Trig2 & ~Trig1: [0, 1, 0, 0, 0]
    print("Result:", cols3.selections["PassHLT"].arr)
    assert np.all(cols3.selections["PassHLT"].arr == [False, True, False, False, False])

    # Case 4: No Match (Expect ValueError)
    print("--- Test 4: No Match ---")
    config4 = [
        ComplexHLTConfig(
            pattern=Pattern(pattern="SetZ*", mode=PatternMode.GLOB), triggers=["Trig1"]
        )
    ]
    mod4 = ComplexHLT(trigger_config=config4)
    cols4 = MockColumns(
        {"dataset_name": "SetA_Run2018", "era": {"trigger_names": trigger_names}},
        hlt_data,
    )
    try:
        mod4.run(cols4, {})
        print("FAILED: Expected ValueError was not raised")
        sys.exit(1)
    except ValueError as e:
        print(f"Caught expected error: {e}")
        assert "No matching trigger config found" in str(e)

    print("ALL TESTS PASSED")


if __name__ == "__main__":
    test()
