import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting.core_plots import *
import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = ["Skim_QCDInclusive2018"]
compressed = [
    f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400", "2000_1400")
]
uncompressed = [f"signal_312_{p}" for p in ("2000_1400", "1200_400", "1500_900")]
both = compressed + uncompressed
representative = [f"signal_312_{p}" for p in ("2000_1900", "1200_400", "1500_900")]


plotter = Plotter(
    "analyzerout/chargino_reco.pkl", "figures", default_backgrounds=backgrounds
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
for x in plots:
    if len(x) == 2:
        (p,cat),add=x,None
    else:
        p,cat,add=x
    plotter(
        p,
        cat,
        [],
        normalize=True,
        add_name=add,
        scale="linear",
        sig_style="hist",
    )
