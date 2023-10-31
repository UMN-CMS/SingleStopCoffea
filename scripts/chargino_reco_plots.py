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


plotter = Plotter("chargino_reco.pkl", "figures", default_backgrounds=backgrounds)
histos = plotter.histos
plotter(
    "mchi_gen_matched",
    compressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m24_m",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_top_3_no_lead_b",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)



plotter(
    "m24_matching",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m13_m",
    compressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m13_matching",
    compressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_top_3_no_lead_b_matching",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_dr_switched",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_dr_switched_matching",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_dr_switched",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_dr_switched_matching",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_top_2_plus_lead_b",
    compressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "m3_top_2_plus_lead_b_matching",
    compressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "perfect_matching_count",
    compressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
    add_name="compressed"
)
plotter(
    "perfect_matching_count",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
    add_name="uncompressed"
)

plotter(
    "mchi_gen_matched",
    compressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
    add_name="compressed",
)

plotter(
    "mchi_gen_matched",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
    add_name="uncompressed",
)
