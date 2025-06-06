import numpy as np
from pathlib import Path
import os
import pickle
import hist
from hist import Hist
from matplotlib import pyplot as plt
import mplhep as hep
import re
import matplotlib

def add_histograms(histogram_paths):
    summed_hist = None
    for histogram_path in histogram_paths:
        with open(histogram_path, 'rb') as picklefile:
            if summed_hist is not None:
                summed_hist += pickle.load(picklefile) 
            else:
                summed_hist = pickle.load(picklefile)

    
    return summed_hist

def plot_summed_hist(histogram_dir, coupling):
    histogram_paths = list(histogram_dir.rglob("*.pkl"))
    summed_hist = add_histograms(histogram_paths)
    length = len(summed_hist.axes[0])
    year_num = 2018
    energy = 13
    unc_values = np.sqrt(summed_hist.variances())
    plt.style.use(hep.style.CMS)

    msd_centers = summed_hist.axes[2].centers
    pt_centers = summed_hist.axes[1].centers
    for i in range(length):
        fig, ax = plt.subplots()
        loop_bin = summed_hist.axes[0][i]
        htbinstr = "_".join([str(k) for k in loop_bin])
        label = f'HT {htbinstr}\nSignal {coupling} Counts' 
        hep.cms.label(year=year_num,ax=ax,data=True,label=label,com=energy,loc=3)
        summed_hist[i,:,:].plot2d(ax=ax)
        for x_bin in range(len(pt_centers)):
            for msd_bin in range(len(msd_centers)):
                # Auto text color based on background
                text_color = 'white'
                sup = f'{unc_values[i,x_bin,msd_bin]:.2f}'
                sub = sup
                text = f"${summed_hist[i,x_bin,msd_bin].value:.2f}^{{{sup}}}_{{{sub}}}$"
                ax.text(pt_centers[x_bin], msd_centers[msd_bin], text,
                    ha="center", va="center",
                    color=text_color, fontsize=12) 
        fig.tight_layout()
        output_title = f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/2018/signal_{coupling}_{htbinstr}.pdf"
        fig.savefig(output_title)
        plt.close()
    
    fig, ax = plt.subplots()
    label = f'Signal {coupling} Counts'
    hep.cms.label(year=year_num,ax=ax,data=True,label=label,com=energy,loc=3)
    summed_hist.project(1,2).plot2d(ax=ax)

    for x_bin in range(len(pt_centers)):
        for msd_bin in range(len(msd_centers)):
            # Auto text color based on background
            text_color = 'white'
            sup = f'{np.sqrt(summed_hist.project(1,2)[x_bin,msd_bin].variance):.2f}'
            sub = sup
            text = f"${summed_hist.project(1,2)[x_bin,msd_bin].value:.2f}^{{{sup}}}_{{{sub}}}$"
            ax.text(pt_centers[x_bin], msd_centers[msd_bin], text,
                ha="center", va="center",
                color=text_color, fontsize=12) 
    fig.tight_layout()
    output_title = f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/2018/signal_{coupling}_summed_ht.pdf"
    fig.savefig(output_title)
    plt.close()

for coupling in ['312','313']:
    histogram_dir = Path(f'/srv/postprocessing/trigger_eff_3d_hist_test/plots/2018/Signal{coupling}')
    plot_summed_hist(histogram_dir=histogram_dir,coupling=coupling)