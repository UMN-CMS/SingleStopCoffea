import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting import simple_plot
import warnings
import itertools as it
import pickle 

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = ["Skim_QCDInclusive2018"]
# compressed = [
#     f"signal_312_{p}" for p in ("1500_1900", "1200_1100", "1500_1400", "2000_1700")
# ]
# uncompressed = [
#     f"signal_312_{p}" for p in ("2000_900", "1200_400", "1500_600", "1500_400")
# ]
# both = compressed + uncompressed

# dense_2000 = [f"signal_312_2000_{p}" for p in (1900, 1700, 1600, 1500, 1300, 1200, 900)]
# dense_1500 = [
#     f"signal_312_1500_{p}" for p in (1450, 1350, 1300, 1200, 1100, 1000, 900, 600, 400)
# ]
# dense_1200 = [f"signal_312_1200_{p}" for p in (1100, 1000, 900, 800, 700, 600, 400)]

# representative = [
#     f"signal_312_{p}"
#     for p in ("2000_1900", "1200_400", "1500_900", "1500_1400", "1200_1100", "2000_900")
# ]

# plotter = Plotter(
#     "myoutput.pkl",
#     "figures",
#     default_backgrounds=None,
#     #default_axis_opts={"number_jets": Plotter.Split},
#     coupling="312",
# )
# plotter("m14_m", ["signal_312_1500_400" , "signal_312_1500_400_mg"],add_label="Pythia vs MG (black)",add_name="400_",normalize=True,ratio=True)
# plotter("m14_m", ["signal_312_1500_1100" , "signal_312_1500_1100_mg"],add_label="Pythia vs MG (black)",add_name="1100_",normalize=True,ratio=True)
# plotter("m14_m", ["signal_312_1500_1400" , "signal_312_1500_1400_mg"],add_label="Pythia vs MG (black)",add_name="1400_",normalize=True, ratio=True)
with (open("myoutput.pkl", "rb")) as openfile:
    objects = pickle.load(openfile)
print(objects.results.keys())
#plotter("m14_vs_m13", ["signal_312_1500_1400_mg"])

sys.exit()


plotter = Plotter(
    "analyzerout/chargino_reco.pkl",
    "figures",
    default_backgrounds=backgrounds,
    default_axis_opts={"number_jets": Plotter.Split},
)
histos = plotter.histos

# plotter("m14_vs_m13", compressed)
# plotter("m14_vs_m24", uncompressed)
plotter("ratio_m14_vs_m13", compressed)
# plotter("ratio_m14_vs_m3_top_3_no_lead_b", uncompressed)
# plotter("m14_vs_m3_top_3_no_lead_b", uncompressed)
# plotter(
#    "m24_m",
#    uncompressed,
#    normalize=False,
#    scale="log",
#    sig_style="hist",
# )
#
# plotter(
#    "m14_m",
#    representative,
#    normalize=False,
#    add_name="all",
#    scale="log",
#    sig_style="hist",
# )
#
# plotter(
#    "m13_m",
#    compressed,
#    normalize=False,
#    scale="log",
#    sig_style="hist",
# )
#
# plotter(
#    "m3_top_3_no_lead_b",
#    uncompressed,
#    normalize=False,
#    scale="log",
#    sig_style="hist",
# )


to_plot = [
    ("max_chi_child_dr", compressed, "compressed"),
    ("max_chi_child_dr", representative, "representative"),
    ("max_chi_child_dr", uncompressed, "uncompressed"),
    ("mean_chi_child_dr", compressed, "compressed"),
    ("mean_chi_child_dr", representative, "representative"),
    ("mean_chi_child_dr", uncompressed, "uncompressed"),
    ("min_chi_child_dr", compressed, "compressed"),
    ("min_chi_child_dr", representative, "representative"),
    ("min_chi_child_dr", uncompressed, "uncompressed"),
    ("max_chi_child_dr", dense_2000, "dense_2000"),
    ("mean_chi_child_dr", dense_2000, "dense_2000"),
    ("min_chi_child_dr", dense_2000, "dense_2000"),
    ("max_chi_child_dr", dense_1500, "dense_1500"),
    ("mean_chi_child_dr", dense_1500, "dense_1500"),
    ("min_chi_child_dr", dense_1500, "dense_1500"),
    ("max_chi_child_dr", dense_1200, "dense_1200"),
    ("mean_chi_child_dr", dense_1200, "dense_1200"),
    ("min_chi_child_dr", dense_1200, "dense_1200"),
    ("max_chi_child_eta", dense_2000, "dense_2000"),
    ("mean_chi_child_eta", dense_2000, "dense_2000"),
    ("min_chi_child_eta", dense_2000, "dense_2000"),
    ("max_chi_child_eta", dense_1500, "dense_1500"),
    ("mean_chi_child_eta", dense_1500, "dense_1500"),
    ("min_chi_child_eta", dense_1500, "dense_1500"),
    ("max_chi_child_eta", dense_1200, "dense_1200"),
    ("mean_chi_child_eta", dense_1200, "dense_1200"),
    ("min_chi_child_eta", dense_1200, "dense_1200"),
    ("max_chi_child_phi", dense_2000, "dense_2000"),
    ("mean_chi_child_phi", dense_2000, "dense_2000"),
    ("min_chi_child_phi", dense_2000, "dense_2000"),
    ("max_chi_child_phi", dense_1500, "dense_1500"),
    ("mean_chi_child_phi", dense_1500, "dense_1500"),
    ("min_chi_child_phi", dense_1500, "dense_1500"),
    ("max_chi_child_phi", dense_1200, "dense_1200"),
    ("mean_chi_child_phi", dense_1200, "dense_1200"),
    ("min_chi_child_phi", dense_1200, "dense_1200"),
    ("max_chi_child_dr_all_three", compressed, "compressed"),
    ("max_chi_child_dr_all_three", representative, "representative"),
    ("max_chi_child_dr_all_three", uncompressed, "uncompressed"),
    ("mean_chi_child_dr_all_three", compressed, "compressed"),
    ("mean_chi_child_dr_all_three", representative, "representative"),
    ("mean_chi_child_dr_all_three", uncompressed, "uncompressed"),
    ("min_chi_child_dr_all_three", compressed, "compressed"),
    ("min_chi_child_dr_all_three", representative, "representative"),
    ("min_chi_child_dr_all_three", uncompressed, "uncompressed"),
    ("max_chi_child_eta", compressed, "compressed"),
    ("max_chi_child_eta", representative, "representative"),
    ("max_chi_child_eta", uncompressed, "uncompressed"),
    ("mean_chi_child_eta", compressed, "compressed"),
    ("mean_chi_child_eta", representative, "representative"),
    ("mean_chi_child_eta", uncompressed, "uncompressed"),
    ("min_chi_child_eta", compressed, "compressed"),
    ("min_chi_child_eta", representative, "representative"),
    ("min_chi_child_eta", uncompressed, "uncompressed"),
    ("max_chi_child_phi", compressed, "compressed"),
    ("max_chi_child_phi", representative, "representative"),
    ("max_chi_child_phi", uncompressed, "uncompressed"),
    ("mean_chi_child_phi", compressed, "compressed"),
    ("mean_chi_child_phi", representative, "representative"),
    ("mean_chi_child_phi", uncompressed, "uncompressed"),
    ("min_chi_child_phi", compressed, "compressed"),
    ("min_chi_child_phi", representative, "representative"),
    ("min_chi_child_phi", uncompressed, "uncompressed"),
    ("stop_pt", representative, "representative"),
]

for n, s, a in to_plot:
    plotter(
        n,
        s,
        [],
        add_name=a,
        normalize=True,
        scale="linear",
        sig_style="hist",
    )
