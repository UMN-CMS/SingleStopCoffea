import awkward as ak
import numpy as np
from analyzer.modules.singlestop.bnn_trig import TriggerBNNCorrection
from analyzer.core.columns import Column


def test_bnn_correction():
    # Mock metadata
    metadata = {"era": {"name": "2016_postVFP"}}

    # Mock data
    ht_data = [500.0, 800.0, 1200.0]
    fj_pt_data = [[400.0], [600.0], [900.0]]  # One fatjet per event

    columns = {
        "HT": ak.Array(ht_data),
        "GoodFatJet": ak.Array([{"pt": pt} for pt in fj_pt_data]),
        "weights": {},
    }

    class MockColumns(dict):
        def __init__(self, data, meta):
            super().__init__(data)
            self.metadata = meta

    cols = MockColumns(columns, metadata)

    # Initialize Module
    base_dir = "/uscms/home/ckapsiak/nobackup/Analysis/SingleStop/SingleStopCoffea"
    pattern = "bnn_correction_{era}.json.gz"

    module = TriggerBNNCorrection(base_path=base_dir, correction_pattern=pattern)

    # Test Variations
    variations = ["central", "up", "down"]

    print(f"Testing TriggerBNNCorrection with file: {base_dir}/{pattern}")
    print(f"Inputs: HT={ht_data}, FatJetPt={[x[0] for x in fj_pt_data]}")

    for var in variations:
        print(f"\n--- Variation: {var} ---")
        params = {"variation": var}
        out_cols, _ = module.run(cols, params)

        weight_col = Column(("Weights", "trigger_eff"))
        weights = out_cols[weight_col]

        print(f"Weights: {weights}")

    print("\nVerification Complete.")


if __name__ == "__main__":
    test_bnn_correction()
