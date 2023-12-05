import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting.core_plots import *
import warnings
import itertools as it

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = ["Skim_QCDInclusive2018"]
compressed = [
    f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400", "2000_1400")
]
uncompressed = [f"signal_312_{p}" for p in ("2000_1400", "1200_400", "1500_900")]
both = compressed + uncompressed
representative = [f"signal_312_{p}" for p in ("2000_1900", "1200_400", "1500_900", "1500_1400", "1200_1100", "2000_900")]


plotter = Plotter("analyzerout/hists.pkl", "figures", default_backgrounds=backgrounds)
histos = plotter.histos
plotter(
    "min_chi_child_dr",
    compressed,
    [],
    normalize=True,
    scale="linear",
    add_name="compressed",
    sig_style="hist",
 )
plotter(
    "max_chi_child_dr",
    compressed,
    [],
    normalize=True,
    scale="linear",
    add_name="compressed",
    sig_style="hist",
 )

plotter(
    "mean_chi_child_dr",
    compressed,
    [],
    normalize=True,
    scale="linear",
    add_name="compressed",
    sig_style="hist",
 )

plotter(
    "stop_pt",
    representative,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
 )

plotter(
    "min_chi_child_dr",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    add_name="uncompressed",
    sig_style="hist",
 )
plotter(
    "max_chi_child_dr",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    add_name="uncompressed",
    sig_style="hist",
 )

plotter(
    "mean_chi_child_dr",
    uncompressed,
    [],
    normalize=True,
    scale="linear",
    add_name="uncompressed",
    sig_style="hist",
 )

# plotter(
#    "m24_m",
#    uncompressed,
#    [],
#    normalize=True,
#    scale="linear",
#    sig_style="hist",
# )
#
# plotter(
#    "m3_top_3_no_lead_b",
#    uncompressed,
#    [],
#    normalize=True,
#    scale="linear",
#    sig_style="hist",
# )


# h = histos["m14_m"]
# nh = h["signal_312_2000_1900", ...]
# dh = h["signal_312_2000_1400", ...]
# plotter.plotPulls("m14_m", "signal_312_2000_1900", "signal_312_2000_1400")
# plotter.plotRatio("m14_m", "signal_312_2000_1900", "signal_312_2000_1400")
