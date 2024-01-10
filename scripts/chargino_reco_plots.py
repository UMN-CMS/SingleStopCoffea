import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting.core_plots import *
import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")


backgrounds = ["Skim_QCDInclusive2018"]
compressed = [
    f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400", "2000_1700")
]
uncompressed = [
    f"signal_312_{p}" for p in ("2000_900", "1200_400", "1500_600", "1500_400")
]
both = compressed + uncompressed

dense_2000 = [f"signal_312_2000_{p}" for p in (1900, 1700, 1600, 1500, 1300, 1200, 900)]
dense_1500 = [
    f"signal_312_1500_{p}" for p in (1450, 1350, 1300, 1200, 1100, 1000, 900, 600, 400)
]
dense_1200 = [f"signal_312_1200_{p}" for p in (1100, 1000, 900, 800, 700, 600, 400)]

representative = [
    f"signal_312_{p}"
    for p in ("2000_1900", "1200_400", "1500_900", "1500_1400", "1200_1100", "2000_900")
]


plotter = Plotter(
    "analyzerout/chargino_reco.pkl",
    "figures",
    default_backgrounds=backgrounds,
    default_axis_opts={"number_jets": Plotter.Split},
    #default_axis_opts={"number_jets": sum},
)
histos = plotter.histos
plots = [
    ("mchi_gen_matched", compressed, "compressed"),
    ("mchi_gen_matched", uncompressed, "uncompressed"),
    ("perfect_matching_count", compressed, "compressed"),
    ("perfect_matching_count", uncompressed, "uncompressed"),
    #
    ("m13_m", compressed),
    ("m3_top_2_plus_lead_b", compressed),
    ("m13_matching", compressed),
    ("m3_top_2_plus_lead_b_matching", compressed),
    ("m13_matching_all_three", compressed),
    ("m3_top_2_plus_lead_b_matching_all_three", compressed),
    #
    ("m24_m", uncompressed),
    ("m3_top_3_no_lead_b", uncompressed),
    ("m3_dr_switched", uncompressed),
    ("m3_top_3_no_lead_b_delta_r_cut", uncompressed),
    ("m24_matching", uncompressed),
    ("m3_top_3_no_lead_b_matching", uncompressed),
    ("m3_dr_switched_matching", uncompressed),
    ("m3_top_3_no_lead_b_dr_cut_matching", uncompressed),
    ("m24_matching_all_three", uncompressed),
    ("m3_top_3_no_lead_b_matching_all_three", uncompressed),
    ("m3_dr_switched_matching_all_three", uncompressed),
    ("m3_top_3_no_lead_b_dr_cut_matching_all_three", uncompressed),
]

plots = [
    ("m14_m", representative),
    ("m14_gt100_m15", representative),
    ]

# for x in plots:
#    if len(x) == 2:
#        (p,cat),add=x,None
#    else:
#        p,cat,add=x
#    plotter(
#        p,
#        cat,
#        [],
#        normalize=True,
#        add_name=add,
#        scale="linear",
#        sig_style="hist",
#    )

#plots = [
#    ("m13_matching_all_three", compressed),
#    ("m24_matching_all_three", uncompressed),
#    ("m3_top_2_plus_lead_b_matching", compressed),
#    ("m3_top_2_plus_lead_b_matching_all_three", compressed),
#    ("m3_top_3_no_lead_b_matching_all_three", uncompressed),
#    ("m3_dr_switched_matching_all_three", uncompressed),
#    ("m3_top_3_no_lead_b_dr_cut_matching_all_three", uncompressed),
#]

for x in plots:
    if len(x) == 2:
        (p, cat), add = x, None
    else:
        p, cat, add = x
    plotter(
        p,
        cat,
        [],
        normalize=True,
        add_name=add,
        scale="linear",
        sig_style="hist",
        xlabel_override="Number Matched",
    )
