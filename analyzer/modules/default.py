import awkward as ak
from analyzer.core import analyzerModule

@analyzerModule("apply_selection", categories="post_selection", always=True)
def applySelection(events, analyzer):
    events = analyzer.applySelection(events)

    hlt_names = analyzer.profile.hlt
    hlt_names = ' | '.join(hlt_names)
    cut_dict = {"ht1200": "HT ≥ 1200", "highptjet": "Jet-PT ≥ 300",
        "jets": "4 ≤ N-Jets ≤ 6", "0Lep": "0e, 0μ", "0looseb": "0b",
        "2bjet": "Med-b Jets ≥ 2", "1tightbjet": "Tight-b Jets ≥ 1",
        "b_dr": "b-jet ΔR > 1", "bbpt": "b-jet 1+2 > 200", "hlt": hlt_names}

    used_cuts = [cut_dict[i] for i in analyzer.selection.names]

    analyzer.dask_result.set_cut_list(used_cuts)

    return events, analyzer
