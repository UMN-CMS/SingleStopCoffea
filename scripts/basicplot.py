import sys
import warnings
sys.path.append(".")

import matplotlib as mpl
from analyzer.plotting.simple_plot import Plotter
mpl.use("Agg")

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = []
sig = [
    f"signal_312_{p}"
    for p in (
        "1000_400",
        "1000_600",
        "1000_900",
        "1200_400",
        "1200_600",
        "1200_1100",
        "1400_400",
        "1400_600",
        "1400_1300",
        "1500_400",
        "1500_600",
        "1500_900",
        "1500_1400",
        "2000_400",
        "2000_600",
        "2000_900",
        "2000_1400",
        "2000_1900",
        "1000_700",
        "1000_800",
        "1200_700",
        "1200_800",
        "1200_900",
        "1200_1000",
        "1500_1000",
        "1500_1100",
        "1500_1200",
        "1500_1300",
        "1500_1350",
        "1500_1450",
        "2000_1200",
        "2000_1300",
        "2000_1500",
        "2000_1600",
        "2000_1700",
    )
]


plotter = Plotter("results/everything.pkl", "plots", default_backgrounds=backgrounds)


toplot = [
    "h_njet",
    "pt_1",
    "pt_2",
    "m14_m",
    "m14_vs_m24",
    "m14_vs_m13",
    "ratio_m14_vs_m24",
    "ratio_m14_vs_m13",
]

for p in toplot:
    plotter(p, sig, normalize=True)
