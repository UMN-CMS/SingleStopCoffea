import awkward as ak
from analyzer.core import analyzerModule
import functools
import operator as op
from .axes import *
from .btag_points import getBTagWP


@analyzerModule("run3_hists", categories="main")
def run3Hists(events, analyzer):
    filled_jets = ak.pad_none(events.good_jets, 4, axis = 1) 

    analyzer.H(
        "HT_presel",
        ht_axis,
        events.HT,
    )

    if '2018' in analyzer.profile.name:
        sorted_bs = ak.pad_none(ak.sort(filled_jets.btagDeepFlavB, axis = 1), 2, axis = 1)
    elif '2023' in analyzer.profile.name:
        sorted_bs = ak.pad_none(ak.sort(filled_jets.btagPNetB, axis = 1), 2, axis = 1)

    pt3_trig = ak.fill_none(filled_jets[:, 2].pt > 35, False)
    pt4_trig = ak.fill_none(filled_jets[:, 3].pt > 30, False)
    b_trig = ak.fill_none(sorted_bs[:, -1] + sorted_bs[:, -2] > 1.1, False)
    trig = pt3_trig & pt4_trig & b_trig

    gj = events.good_jets[trig]
    med_bs = events.med_bs[trig]
    good_bs = sorted_bs[trig]
    tight_bs = events.tight_bs[trig]

    analyzer.H(
        "m3_comp",
        makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV"),
        gj[:,0:3].sum().mass,
        mask = trig,
    )

    analyzer.H(
        "m3_uncomp",
        makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV"),
        gj[:,1:4].sum().mass,
        mask = trig,
    )

    analyzer.H(
        "m4",
        makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
        gj[:, 0:4].sum().mass,
        mask = trig,
    )

    analyzer.H(
        "ratio_m3_comp_m4",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig,
    )

    analyzer.H(
        "ratio_m3_uncomp_m4",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig,
    )

    analyzer.H(
        "pt_0",
        makeAxis(50, 0, 1000, "$p_{T, 0}$", unit="GeV"),
        gj[:,0].pt,
        mask = trig,
    )

    analyzer.H(
        "pt_1",
        makeAxis(50, 0, 1000, "$p_{T, 1}$", unit="GeV"),
        gj[:,1].pt,
        mask = trig,
    )

    analyzer.H(
        "pt_2",
        makeAxis(50, 0, 1000, "$p_{T, 2}$", unit="GeV"),
        gj[:,2].pt,
        mask = trig,
    )

    analyzer.H(
        "pt_3",
        makeAxis(50, 0, 1000, "$p_{T, 3}$", unit="GeV"),
        gj[:,3].pt,
        mask = trig,
    )

    analyzer.H(
        "HT",
        ht_axis,
        events[trig].HT,
        mask = trig,
    )

    analyzer.H(
        "nb_med",
        makeAxis(5, 0, 5, "$n_{b}$ (medium)"),
        ak.num(med_bs, axis=1),
        mask = trig,
    )     

    analyzer.H(
        "nb_tight",
        makeAxis(5, 0, 5, "$n_{b}$ (tight)"),
        ak.num(tight_bs, axis=1),
        mask = trig,
    )     

    analyzer.H(
        "nj",
        nj_axis,
        ak.num(gj, axis=1),
        mask = trig,
    )

    med_b_mask = ak.fill_none(ak.num(events.med_bs, axis = 1) >= 2, False)
    med_bs = events.med_bs[trig & med_b_mask]

    analyzer.H(
        "dRbb_01",
        makeAxis(20, 0, 5, "|$\Delta R_{{b_0}, {b_1}}$|"),
        abs(med_bs[:, 0].delta_r(med_bs[:, 1])),
        mask = trig & med_b_mask,
    )

    # Cuts
    
    # Number of jets
    nj_cut = (ak.fill_none((ak.num(events.good_jets, axis = 1) >= 4), False)) & (ak.fill_none((ak.num(events.good_jets, axis = 1) <= 5), False))
    gj = events.good_jets[trig & nj_cut]
    analyzer.H(
        "nj_cut_gte4_lte5_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & nj_cut,
    )

    analyzer.H(
        "nj_cut_gte4_lte5_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & nj_cut,
    )

    nj_cut = (ak.fill_none((ak.num(events.good_jets, axis = 1) >= 4), False)) & (ak.fill_none((ak.num(events.good_jets, axis = 1) <= 6), False))
    gj = events.good_jets[trig & nj_cut]
    analyzer.H(
        "nj_cut_gte4_lte6_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & nj_cut,
    )

    analyzer.H(
        "nj_cut_gte4_lte6_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & nj_cut,
    )

    nj_cut = (ak.fill_none((ak.num(events.good_jets, axis = 1) >= 4), False)) & (ak.fill_none((ak.num(events.good_jets, axis = 1) <= 7), False))
    gj = events.good_jets[trig & nj_cut]
    analyzer.H(
        "nj_cut_gte4_lte7_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & nj_cut,
    )

    analyzer.H(
        "nj_cut_gte4_lte7_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & nj_cut,
    )

    nj_cut = (ak.fill_none((ak.num(events.good_jets, axis = 1) >= 4), False)) & (ak.fill_none((ak.num(events.good_jets, axis = 1) <= 8), False))
    gj = events.good_jets[trig & nj_cut]
    analyzer.H(
        "nj_cut_gte4_lte8_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & nj_cut,
    )

    analyzer.H(
        "nj_cut_gte4_lte8_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & nj_cut,
    )

    # Number of bs
    nb_cut = med_b_mask & (ak.fill_none(ak.num(events.tight_bs, axis = 1) >= 0, False))
    gj = events.good_jets[trig & nb_cut]
    analyzer.H(
        "nb_cut_2med_0tight_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & med_b_mask & nb_cut,
    )

    analyzer.H(
        "nb_cut_2med_0tight_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & med_b_mask & nb_cut,
    )

    nb_cut = med_b_mask & (ak.fill_none(ak.num(events.tight_bs, axis = 1) >= 1, False))
    gj = events.good_jets[trig & nb_cut]
    analyzer.H(
        "nb_cut_2med_1tight_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & med_b_mask & nb_cut,
    )

    analyzer.H(
        "nb_cut_2med_1tight_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & med_b_mask & nb_cut,
    )

    nb_cut = med_b_mask & (ak.fill_none(ak.num(events.tight_bs, axis = 1) >= 2, False))
    gj = events.good_jets[trig & nb_cut]
    analyzer.H(
        "nb_cut_2med_2tight_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & med_b_mask & nb_cut,
    )

    analyzer.H(
        "nb_cut_2med_2tight_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & med_b_mask & nb_cut,
    )

    # HT
    HT_cut = (events.HT > 400)
    gj = events.good_jets[trig & HT_cut]
    analyzer.H(
        "HT_cut_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & HT_cut,
    )

    analyzer.H(
        "HT_cut_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & HT_cut,
    )

    # dRbb
    filled_bs = ak.pad_none(events.med_bs, 2, axis = 1)
    dRbb_cut = (ak.fill_none(filled_bs[:, 0].delta_r(filled_bs[:, 1]) > 1, False))
    gj = events.good_jets[trig & dRbb_cut]
    analyzer.H(
        "dRbb_cut_gt1_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & dRbb_cut,
    )

    analyzer.H(
        "dRbb_cut_gt1_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & dRbb_cut,
    )

    filled_bs = ak.pad_none(events.med_bs, 2, axis = 1)
    dRbb_cut = (ak.fill_none(filled_bs[:, 0].delta_r(filled_bs[:, 1]) > 1, False)) & (ak.fill_none(filled_bs[:, 0].delta_r(filled_bs[:, 1]) < 3.2, False)) 
    gj = events.good_jets[trig & dRbb_cut]
    analyzer.H(
        "dRbb_cut_gt1_lt32_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & dRbb_cut,
    )

    analyzer.H(
        "dRbb_cut_gt1_lt32_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & dRbb_cut,
    )

    # Leading jet pT
    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 100, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_100_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_100_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )

    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 120, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_120_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_120_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )

    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 140, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_140_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_140_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )

    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 160, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_160_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_160_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )

    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 180, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_180_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_180_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )

    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 200, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_200_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_200_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )

    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 220, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_220_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_220_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )

    filled_jets = ak.pad_none(events.good_jets, 1, axis = 1)
    pt_cut = ak.fill_none(filled_jets[:, 0].pt > 240, False)
    gj = events.good_jets[trig & pt_cut]
    analyzer.H(
        "pt_cut_240_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut,
    )

    analyzer.H(
        "pt_cut_240_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut,
    )
    # All Cuts
    gj = events.good_jets[trig & nb_cut & HT_cut & dRbb_cut & pt_cut]
    analyzer.H(
        "all_cuts_comp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (comp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 0:3].sum().mass],
        mask = trig & pt_cut & nb_cut & HT_cut & dRbb_cut & pt_cut,
    )

    analyzer.H(
        "all_cuts_uncomp",
        [ makeAxis(60, 0, 3000, "$m_{4}$", unit="GeV"),
          makeAxis(60, 0, 3000, "$m_{3}$ (uncomp.)", unit="GeV")
        ],
        [gj[:, 0:4].sum().mass, gj[:, 1:4].sum().mass],
        mask = trig & pt_cut & nb_cut & HT_cut & dRbb_cut & pt_cut,
    )

    return events, analyzer
