import sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
import mplhep as hep

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
        #"200_100",
        #"300_100",
        #"300_200",
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

compressed = [f"signal_312_{f}" for f in
             ( "200_100",
               "300_200",
               "500_400",
               "700_600",
               "900_800",
               "1000_900",
               "1200_1100",
               "1500_1400",
               "2000_1900",
              )
]


uncompressed = [f"signal_312_{f}" for f in
             ( "500_100",
               "700_100",
               "900_400",
               "1000_400",
               "1200_700",
               "1500_900",
               "2000_1200",
              )
]

signal_vals = {s : np.zeros(2) for s in sig}

backgrounds = ["QCDInclusive2023", "TTToHadronic2018"]

'''
signal_survival_rates = {(int(s.split('_')[2]), int(s.split('_')[3])) : 0 for s in sig}

for sample in (sig):
    d = pkl.load(open(f"Run3/{sample}.pkl", "rb"))

    profile_repo = ds.ProfileRepo()
    profile_repo.loadFromDirectory("profiles")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory('datasets/', profile_repo)
    hists = d.getMergedHistograms(sample_manager)

    p = 'ratio_m3_comp_m4'
    m3m4_list = np.array(hists[p][f'{sample}']).tolist()
    m3m4_vals = np.array([[t[0] for t in row] for row in m3m4_list])
    signal_vals[sample] = m3m4_vals

    p = 'HT_presel'
    HT_presel_list = np.array(hists[p][f'{sample}']).tolist()
    HT_presel_vals = np.array([t[0] for t in HT_presel_list])
    

    p = 'HT'
    HT_list = np.array(hists[p][f'{sample}']).tolist()
    HT_vals = np.array([t[0] for t in HT_list])

    signal_survival_rates[(int(sample.split('_')[2]), int(sample.split('_')[3]))] = np.sum(HT_vals) / np.sum(HT_presel_vals)    
signal_survival_rates = {s : signal_survival_rates[s] for s in sorted(signal_survival_rates.keys())}

for sample in backgrounds:
    d = pkl.load(open(f"Run3/{sample}.pkl", "rb"))

    profile_repo = ds.ProfileRepo()
    profile_repo.loadFromDirectory("profiles")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory('datasets/', profile_repo)
    hists = d.getMergedHistograms(sample_manager)

    toplot = [h for h in hists.keys() if 'unweighted' not in h]
    
    p = 'ratio_m3_comp_m4'
    significances = {s : 0 for s in sorted(signal_survival_rates.keys())}
    m3m4_list = np.array(hists[p][f'{sample}']).tolist()
    m3m4_vals = np.array([[t[0] for t in row] for row in m3m4_list])
    for s in signal_vals:
        significance_arr = np.divide(signal_vals[s][m3m4_vals > 0], np.sqrt(m3m4_vals[m3m4_vals > 0]))
        significances[(int(s.split('_')[2]), int(s.split('_')[3]))] = np.sqrt(np.sum(np.square(significance_arr)))
'''

def stop_chargino_plots(data_dict, outfile, title = "", cmap = 'Greens', cbar_label = r"$s / \sqrt{b}$ ratio"): 
    hep.style.use("CMS")
    rows, cols = np.meshgrid(np.arange(500, 2050, 100), np.arange(100, 1950, 100))
    
    fig, ax = plt.subplots()
    data = np.zeros_like(rows, dtype = float)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if (rows[i, j], cols[i, j]) in data_dict:
                data[i, j] = data_dict[(rows[i, j], cols[i, j])]

    data[data == 0] = np.nan

    f1 = ax.pcolormesh(rows, cols, data, cmap = cmap)
    fig.colorbar(f1, ax=ax, label = cbar_label)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            text = str(round(data[i, j], 2))
            if data[i, j] > 0: plt.text(rows[i, j] - 25, cols[i, j], text, fontsize = 8)

    plt.xticks(np.arange(500, 2050, 500), rotation = 90)
    plt.yticks(np.arange(100, 1950, 500))

    ax.set_xlabel(r'$m_{\tilde{t}}$ [GeV]')    
    ax.set_ylabel(r'$m_{\tilde{\chi}^{\pm}_{1}}$ [GeV]')    
    ax.set_title(title, fontsize = 'medium')

    fig.tight_layout()
    plt.savefig(f'plots/{outfile}.pdf')
    return fig, ax

def calculate_significance(cut_hist):
    signal_vals = {(int(s.split('_')[2]), int(s.split('_')[3])) : 0 for s in sig}
    signal_m3m4_vals = {(int(s.split('_')[2]), int(s.split('_')[3])) : 0 for s in sig}
    cut_significances = {(int(s.split('_')[2]), int(s.split('_')[3])) : 0 for s in sig}

    for sample in sig:
        print(sample)
        d = pkl.load(open(f"Run3/{sample}.pkl", "rb"))
        profile_repo = ds.ProfileRepo()
        profile_repo.loadFromDirectory("profiles")
        sample_manager = ds.SampleManager()
        sample_manager.loadSamplesFromDirectory('datasets/', profile_repo)
        hists = d.getMergedHistograms(sample_manager)

        cut_hist_list = np.array(hists[cut_hist][f'{sample}']).tolist()
        cut_hist_vals = np.array([[t[0] for t in row] for row in cut_hist_list])
        signal_vals[(int(sample.split('_')[2]), int(sample.split('_')[3]))] = cut_hist_vals
            
        m3m4_list = np.array(hists['ratio_m3_comp_m4'][f'{sample}']).tolist()
        m3m4_vals = np.array([[t[0] for t in row] for row in m3m4_list])
        signal_m3m4_vals[(int(sample.split('_')[2]), int(sample.split('_')[3]))] = m3m4_vals

        if 'HT' in cut_hist: print(sample, np.sum(cut_hist_vals), np.sum(m3m4_vals))

    cut_hist_vals = np.zeros_like(cut_hist_vals)
    cut_hist_var = np.zeros_like(cut_hist_vals)
    m3m4_vals = np.zeros_like(m3m4_vals)
    m3m4_var = np.zeros_like(m3m4_vals)

    for sample in backgrounds:
        d = pkl.load(open(f"Run3/{sample}.pkl", "rb"))
        hists = d.getMergedHistograms(sample_manager)

        cut_hist_list = np.array(hists[cut_hist][f'{sample}']).tolist()
        cut_hist_vals += np.array([[t[0] for t in row] for row in cut_hist_list])
        cut_hist_var += np.array([[t[1] for t in row] for row in cut_hist_list])
            
        m3m4_list = np.array(hists['ratio_m3_comp_m4'][f'{sample}']).tolist()
        m3m4_vals += np.array([[t[0] for t in row] for row in m3m4_list])
        m3m4_var += np.array([[t[1] for t in row] for row in m3m4_list])

        
        p = 'HT_presel'
        HT_presel_list = np.array(hists[p][f'{sample}']).tolist()
        HT_presel_vals = np.array([t[0] for t in HT_presel_list])
    

        p = 'HT'
        HT_list = np.array(hists[p][f'{sample}']).tolist()
        HT_vals = np.array([t[0] for t in HT_list])

        #print(np.sum(HT_vals), np.sum(HT_presel_vals))
        if 'HT' in cut_hist: print(sample, np.sum(cut_hist_vals), np.sum(m3m4_vals))

    cut_out_bkg_events = m3m4_vals - cut_hist_vals

    for s in signal_vals:
        cut_hist_sig_arr = np.sqrt(2 * ((signal_vals[s] + cut_hist_vals) * np.log(1 + np.divide(signal_vals[s], cut_hist_vals, out=np.zeros_like(cut_hist_vals), where=cut_hist_vals!=0)) - signal_vals[s]))
        #cut_hist_sig_arr = np.divide(signal_vals[s], np.sqrt(cut_hist_vals + cut_hist_var), out=np.zeros_like(cut_hist_vals), where=cut_hist_vals!=0)
        cut_hist_sig = np.sqrt(np.sum(np.square(cut_hist_sig_arr)))

        baseline_sig_arr = np.sqrt(2 * ((signal_m3m4_vals[s] + m3m4_vals) * np.log(1 + np.divide(signal_m3m4_vals[s], m3m4_vals, out=np.zeros_like(m3m4_vals), where=m3m4_vals!=0)) - signal_m3m4_vals[s]))
        #baseline_sig_arr = np.divide(signal_m3m4_vals[s], np.sqrt(m3m4_vals + m3m4_var, out=np.zeros_like(m3m4_vals), where=m3m4_vals != 0))
        baseline_sig = np.sqrt(np.sum(np.square(baseline_sig_arr)))

        cut_significances[s] = cut_hist_sig / baseline_sig

        hep.style.use("CMS")
        rows, cols = np.meshgrid(np.arange(0, 3050, 50), np.arange(0, 3050, 50))
        fig, ax = plt.subplots()

        f1 = ax.pcolormesh(rows, cols, cut_hist_sig_arr.T)
        fig.colorbar(f1, ax=ax, label = 'Significance')

        plt.xticks(np.arange(0, 3050, 500), rotation = 90)
        plt.yticks(np.arange(0, 3050, 500))
   
        ax.set_xlabel(r'$m_{4}$ [GeV]')    
        ax.set_ylabel(r'$m_{3}$ [GeV]')    
        ax.set_title(f'{cut_hist} {s} Significance', fontsize = 'medium')
   
        fig.tight_layout()
        plt.savefig(f'plots/cut_significances/{s}_{cut_hist}_significance.pdf')
     
        fig, ax = plt.subplots()

        f1 = ax.pcolormesh(rows, cols, baseline_sig_arr.T)
        fig.colorbar(f1, ax=ax, label = 'Significance')

        plt.xticks(np.arange(0, 3050, 500), rotation = 90)
        plt.yticks(np.arange(0, 3050, 500))
   
        ax.set_xlabel(r'$m_{4}$ [GeV]')    
        ax.set_ylabel(r'$m_{3}$ [GeV]')    
        ax.set_title(f'Baseline {s} Significance', fontsize = 'medium')
   
        fig.tight_layout()
        plt.savefig(f'plots/cut_significances/{s}_baseline_significance.pdf')

        fig, ax = plt.subplots()

        cut_out_sig_vals = signal_m3m4_vals[s] - signal_vals[s]
        f1 = ax.pcolormesh(rows, cols, cut_out_sig_vals.T)
        fig.colorbar(f1, ax=ax, label = 'Events')

        plt.xticks(np.arange(0, 3050, 500), rotation = 90)
        plt.yticks(np.arange(0, 3050, 500))
   
        ax.set_xlabel(r'$m_{4}$ [GeV]')    
        ax.set_ylabel(r'$m_{3}$ [GeV]')    
        ax.set_title(f'HT <= 400 Plane', fontsize = 'medium')
   
        fig.tight_layout()
        plt.savefig(f'plots/cut_significances/cut_out_{s}_vals.pdf')

    fig, ax = plt.subplots()
    f1 = ax.pcolormesh(rows, cols, cut_out_bkg_events.T)
    fig.colorbar(f1, ax=ax, label = 'Events')

    plt.xticks(np.arange(0, 3050, 500), rotation = 90)
    plt.yticks(np.arange(0, 3050, 500))
   
    ax.set_xlabel(r'$m_{4}$ [GeV]')    
    ax.set_ylabel(r'$m_{3}$ [GeV]')    
    ax.set_title(f'HT <= 400 Plane', fontsize = 'medium')
   
    fig.tight_layout()
    plt.savefig(f'plots/cut_significances/cut_out_bkg_vals.pdf')
    return cut_significances

#stop_chargino_plots(significances, 'significances_no_cuts', title = r"$S / \sqrt{B}$ After Trigger")
#stop_chargino_plots(signal_survival_rates, 'signal_survival_rates', title = "Signal Trigger Pass Rate")
#stop_chargino_plots(calculate_significance('HT_cut_comp'), 'HT_cut_comp', title = r'$H_T > 400$ Cut')

stop_chargino_plots(calculate_significance('nb_cut_2med_0tight_comp'), 'nb_cut_2med_0tight_comp', title = r'2 medium bs, 0 tight b Cut')
stop_chargino_plots(calculate_significance('nb_cut_2med_1tight_comp'), 'nb_cut_2med_1tight_comp', title = r'2 medium bs, 1 tight b Cut')
stop_chargino_plots(calculate_significance('nb_cut_2med_2tight_comp'), 'nb_cut_2med_2tight_comp', title = r'2 medium bs, 2 tight b Cut')

stop_chargino_plots(calculate_significance('dRbb_cut_gt1_comp'), 'dRbb_cut_gt1_comp', title = r'$\Delta R_{b_0, b_1} > 1$ Cut')
stop_chargino_plots(calculate_significance('dRbb_cut_gt1_lt32_comp'), 'dRbb_cut_gt1_lt32_comp', title = r'$1 < \Delta R_{b_0, b_1} < 3.2$ Cut')

stop_chargino_plots(calculate_significance('pt_cut_100_comp'), 'pt_cut_100_comp', title = r'$p_{T, 0} > 100$ Cut')
stop_chargino_plots(calculate_significance('pt_cut_120_comp'), 'pt_cut_120_comp', title = r'$p_{T, 0} > 120$ Cut')
stop_chargino_plots(calculate_significance('pt_cut_140_comp'), 'pt_cut_140_comp', title = r'$p_{T, 0} > 140$ Cut')

#stop_chargino_plots(calculate_significance('all_cuts_comp'), 'all_cuts_comp', title = r'$H_T > 400$, 2 medium bs, 1 tight b, $\Delta R_{b_0, b_1} > 1$, $p_{T, 0} > 140$')
#
