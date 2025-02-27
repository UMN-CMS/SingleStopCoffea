import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")
import analyzer.datasets as ds
import pickle as pkl

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
        "200_100",
        "300_100",
        "300_200",
        "500_100",
        "500_200",
        "500_400",
        "700_100",
        "700_400",
        "700_600",
        "800_400",
        "800_600",
        "800_700",
        "900_400",
        "900_600",
        "900_700",
        "900_800",
            )
]

#sig = []

backgrounds = ["QCDInclusive2023"]

for sample in (sig + backgrounds):
    plotter = Plotter(f"Run3/{sample}.pkl", f"plots/{sample}/", default_backgrounds=sample)

    d = pkl.load(open(f"Run3/{sample}.pkl", "rb"))

    profile_repo = ds.ProfileRepo()
    profile_repo.loadFromDirectory("profiles")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory('datasets/', profile_repo)
    hists = d.getMergedHistograms(sample_manager)

    '''
    toplot = [
    "h_njet",
    "HT",
    "pt_1",
    "pt_2",
    "m14_m",
    "m13m",
    "m24_vs_m14",
    "m13_vs_m14",
    ]
    '''

    toplot = [h for h in hists.keys() if 'unweighted' not in h]

    for p in toplot:
        plotter(p, [], normalize=True, add_name=f"{sample}")

combined = [f"signal_312_{f}" for f in
           ( "200_100",
             "300_200",
             "500_100",
             "700_400",
             "900_600",
             "1000_900",
             "1500_600",
             "1500_900",
             "2000_1900",
           )
]


plotter = Plotter(f"Run3/combined.pkl", f"plots/combined/", default_backgrounds="QCDInclusive2023")

d = pkl.load(open(f"Run3/combined.pkl", "rb"))

profile_repo = ds.ProfileRepo()
profile_repo.loadFromDirectory("profiles")
sample_manager = ds.SampleManager()
sample_manager.loadSamplesFromDirectory('datasets/', profile_repo)
hists = d.getMergedHistograms(sample_manager)

toplot = [h for h in hists.keys() if 'unweighted' not in h]
for p in toplot:
    plotter(p, combined, normalize=True, default_background = "QCDInclusive2023")

#plotter('d_r_mu_j1', normalize=False, scale="log")
#plotter('d_r_mu_j2', normalize=False, scale="log")
#plotter('d_r_mu_j3', normalize=False, scale="log")
#plotter('d_r_mu_j4', normalize=False, scale="log")
#plotter("pT0_vs_mSoftDrop", [], normalize = False)

#plotter.plotRatio(f'{sample}', 'totalHT', 'passedHT')
#plotter.plotRatio('Run3/QCDInclusive2023', 'total_pt0', 'passed_pt0')
#plotter.plotRatio2D('Run3/QCDInclusive2023', 'total_pT0_vs_mSoftDrop', 'passed_pT0_vs_mSoftDrop')
