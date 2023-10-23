import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting.core_plots import *
import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = ["Skim_QCDInclusive2018"]
compressed = [f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400", "2000_1400")]
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
    "mchi_gen_matched",
    compressed,
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



#h = histos["m14_m"]
#nh = h["signal_312_2000_1900", ...]
#dh = h["signal_312_2000_1400", ...]
#plotter.plotPulls("m14_m", "signal_312_2000_1900", "signal_312_2000_1400")
#plotter.plotRatio("m14_m", "signal_312_2000_1900", "signal_312_2000_1400")
