import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting import simple_plot
import warnings
import itertools as it
import pickle

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

# backgrounds = ["Skim_QCDInclusive2018"]
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

plotter = Plotter(
    "myoutput.pkl",
    "figures",
    default_backgrounds=None,
    #default_axis_opts={"number_jets": Plotter.Split},
    coupling="312",
)
list_of_samples = ["400","400_mg","1100","1100_mg","1400","1400_mg"]
pt_hists = ['m14_pt','m13_pt','m24_pt','m12_pt','m23_pt',]
list_of_mass_points = ["1400"]
list_of_2d_hists = ['m14_vs_m13', 'ratio_m14_vs_m13', 'm14_vs_m24', 'ratio_m14_vs_m24', 'm14_vs_m12', 'ratio_m14_vs_m12', 
                    'm14_vs_m23', 'ratio_m14_vs_m23', 'm13_vs_m24', 'ratio_m13_vs_m24', 'm13_vs_m12', 'ratio_m13_vs_m12', 'm13_vs_m23', 'ratio_m13_vs_m23', 
                    'm24_vs_m12', 'ratio_m24_vs_m12', 'm24_vs_m23', 'ratio_m24_vs_m23', 'm12_vs_m23', 'ratio_m12_vs_m23','d_phi_01_vs_02', 
                    'd_phi_01_vs_03', 'd_phi_01_vs_12', 'd_phi_01_vs_13', 'd_phi_01_vs_23', 'd_phi_02_vs_03', 'd_phi_02_vs_12', 'd_phi_02_vs_13', 
                    'd_phi_02_vs_23', 'd_phi_03_vs_12', 'd_phi_03_vs_13', 'd_phi_03_vs_23', 'd_phi_12_vs_13', 'd_phi_12_vs_23', 'd_phi_13_vs_23']
list_of_1d_hists = ['h_njet', 'm14_pt', 'm14_eta', 'm14_m', 'm13_pt', 'm13_eta', 'm13_m', 'm24_pt', 'm24_eta', 'm24_m', 'm12_pt', 'm12_eta', 'm12_m', 
                    'm23_pt', 'm23_eta', 'm23_m', 'pt_0', 'eta_0', 'phi_0', 'pt_1', 
                    'eta_1', 'phi_1', 'pt_2', 'eta_2', 'phi_2', 'pt_3', 'eta_3', 'phi_3', 'd_eta_1_1', 'd_phi_1_1', 'd_r_1_1', 'd_eta_1_2', 'd_phi_1_2', 
                    'd_r_1_2', 'd_eta_1_3', 'd_phi_1_3', 'd_r_1_3', 'd_eta_1_4', 'd_phi_1_4', 'd_r_1_4', 'd_eta_2_2', 'd_phi_2_2', 'd_r_2_2', 'd_eta_2_3', 
                    'd_phi_2_3', 'd_r_2_3', 'd_eta_2_4', 'd_phi_2_4', 'd_r_2_4', 'd_eta_3_3', 'd_phi_3_3', 'd_r_3_3', 'd_eta_3_4', 'd_phi_3_4', 'd_r_3_4', 
                    'd_eta_4_4', 'd_phi_4_4', 'd_r_4_4', 'pt_ht_ratio_1', 'pt_ht_ratio_2', 'pt_ht_ratio_3', 'pt_ht_ratio_4', 'pt_ht_ratio_5']
for j in pt_hists:
    plotter(j, ["signal_312_1500_400", "signal_312_1500_400_mg"],add_label="Pythia vs MG (black)",add_name="400",normalize=True,ratio=True)
    plotter(j, ["signal_312_1500_1100", "signal_312_1500_1100_mg"],add_label="Pythia vs MG (black)",add_name="1100",normalize=True,ratio=True)
    plotter(j, ["signal_312_1500_1400", "signal_312_1500_1400_mg"],add_label="Pythia vs MG (black)",add_name="1400",normalize=True,ratio=True)

# plotter('m14_pt', ["signal_312_1500_400", "signal_312_1500_400_mg"],add_label="Pythia vs MG (black)",add_name="400",normalize=False,ratio=True)
# plotter('m14_pt', ["signal_312_1500_1100", "signal_312_1500_1100_mg"],add_label="Pythia vs MG (black)",add_name="1100",normalize=False,ratio=True)
# plotter('m14_pt', ["signal_312_1500_1400", "signal_312_1500_1400_mg"],add_label="Pythia vs MG (black)",add_name="1400",normalize=False,ratio=True)
# for j in list_of_2d_hists:
#     plotter(j,["signal_312_1500_400","signal_312_1500_400_mg"],add_label=j,add_name="400",normalize=True,ratio=True)
#     plotter(j,["signal_312_1500_1100","signal_312_1500_1100_mg"],add_label=j,add_name="1100",normalize=True,ratio=True)
#     plotter(j,["signal_312_1500_1400","signal_312_1500_1400_mg"],add_label=j,add_name="1400",normalize=True,ratio=True)
print(plotter.histos.keys())

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
