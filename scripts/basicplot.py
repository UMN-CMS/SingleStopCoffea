import sys
import numpy as np

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

backgrounds = ["QCDInclusive2023", "TTToHadronic2018"]

combined_comp = [f"signal_312_{f}" for f in
           ( "1200_1100",
             "1500_1400",
             "2000_1900",
           )
]


combined_uncomp = [f"signal_312_{f}" for f in
           ( "1200_400",
             "1500_600",
             "2000_1200",
           )
]

compressed = [f"signal_312_{f}" for f in
             ( #"200_100",
               #"300_200",
               "500_400",
               "700_600",
               "900_800",
               "1000_900",
               "1200_1100",
               #"1500_1400",
               #"2000_1900",
              )
]


uncompressed = [f"signal_312_{f}" for f in
             ( "500_100",
               "700_100",
               "900_400",
               "1000_400",
               "1200_700",
               #"1500_900",
               #"2000_1200",
              )
]

toplot = ['HT', 
         'm3_comp',
         'm3_uncomp', 
         'm4', 
         'nb_med', 
         'nb_tight', 
         'nj', 
         'dRbb_01', 
         'pt_0', 
         'pt_1', 
         'pt_2', 
         'pt_3'
]

'''
for sample in (sig + backgrounds):
    plotter = Plotter(f"Run3/{sample}.pkl", f"plots/{sample}/", default_backgrounds=sample)

    for p in toplot:
        plotter(p, [], normalize=True, add_name=f"{sample}")

    for p in ['ratio_m3_comp_m4', 'ratio_m3_uncomp_m4', 'HT_cut_comp', 'HT_cut_uncomp']:
        plotter(p, [], normalize=False, add_name=f"{sample}")
'''

for h in toplot:
    comp_fnames = ["Run3/" + samp + ".pkl" for samp in (compressed + backgrounds)]
    plotter = Plotter(comp_fnames, f"plots/compressed/", default_backgrounds = backgrounds)
    plotter(h, compressed, normalize=True, add_name="compressed")

    uncomp_fnames = ["Run3/" + samp + ".pkl" for samp in (uncompressed + backgrounds)]
    plotter = Plotter(uncomp_fnames, f"plots/uncompressed/", default_backgrounds = backgrounds)
    plotter(h, uncompressed, normalize=True, add_name="uncompressed")

    combined_comp_fnames = ["Run3/" + samp + ".pkl" for samp in (combined_comp + backgrounds)]
    plotter = Plotter(combined_comp_fnames, f"plots/combined_comp/", default_backgrounds = backgrounds)
    #plotter(h, combined_comp, normalize=True, add_name="combined_comp")

    combined_uncomp_fnames = ["Run3/" + samp + ".pkl" for samp in (combined_uncomp + backgrounds)]
    plotter = Plotter(combined_uncomp_fnames, f"plots/combined_uncomp/", default_backgrounds = backgrounds)
    #plotter(h, combined_uncomp, normalize=True, add_name="combined_uncomp")
#plotter('d_r_mu_j1', normalize=False, scale="log")
#plotter('d_r_mu_j2', normalize=False, scale="log")
#plotter('d_r_mu_j3', normalize=False, scale="log")
#plotter('d_r_mu_j4', normalize=False, scale="log")
#plotter("pT0_vs_mSoftDrop", [], normalize = False)

#plotter.plotRatio(f'{sample}', 'totalHT', 'passedHT')
#plotter.plotRatio('Run3/QCDInclusive2023', 'total_pt0', 'passed_pt0')
#plotter.plotRatio2D('Run3/QCDInclusive2023', 'total_pT0_vs_mSoftDrop', 'passed_pT0_vs_mSoftDrop')
