import numpy as np
from pathlib import Path
import os
import pickle
import hist
from hist import Hist
from matplotlib import pyplot as plt
import mplhep as hep
import re

def calculate_scale_factors():
    
    for i in ["2016_preVFP","2016_preVFP","2016_postVFP","2017","2018","2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"]:
        for j in ["SingleJetEff", "HTEff"]:
            ratio_path =  list(Path(f"/srv/postprocessing/trigger_eff/plots/{i}/{j}/").rglob("*.pkl"))
            with open(ratio_path[0], 'rb') as picklefile:
                num_dict = pickle.load(picklefile)
            with open(ratio_path[1], 'rb') as picklefile:
                den_dict = pickle.load(picklefile)
            num_h, num_unc = num_dict["Hist"], num_dict["Unc"]
            den_h, den_unc = den_dict["Hist"], den_dict["Unc"]
            sf_h = num_h/den_h

            # check the directory does not exist
            path = f"/srv/analyzer_resources/datavmc_sf/{i}/"
            fig, axs = plt.subplots(2,1)
            plt.style.use(hep.style.CMS)
            num_values, num_bins = num_h.to_numpy()
            den_values, den_bins = den_h.to_numpy()

            sf_values, sf_bins = sf_h.to_numpy()
            num_mask = np.isnan(num_values)
            den_mask = np.isnan(den_values)
            nan_mask = num_mask | den_mask

            num_mask = ~num_mask
            den_mask = ~den_mask

            num_x = num_bins[1:]-(num_bins[1]-num_bins[0])/2     
            den_x = den_bins[1:]-(den_bins[1]-den_bins[0])/2
            sf_x = sf_bins[1:]-(sf_bins[1]-sf_bins[0])/2
            
            sf_values[nan_mask] = 0
            
            den_h.plot1d(yerr=False,ax=axs[0],label='QCD TE',color='b')
            axs[0].errorbar(den_x[den_mask],den_values[den_mask],den_unc[:,den_mask],linestyle='',color='b')
            axs[0].set_ylabel("Efficiency")
            num_h.plot1d(yerr=False,ax=axs[0],label='Data TE',color='r')
            axs[0].errorbar(num_x[num_mask],num_values[num_mask],num_unc[:,num_mask],linestyle='',color='r')
            import re
            year_num = int(re.findall(r'\d+', i)[0])
            if year_num > 2018:
                energy = 13.6
            else:
                energy = 13
            hep.cms.label(year=i,ax=axs[0],data=True,label="Preliminary",com=energy)
            axs[1].plot(sf_x,sf_values,label='SF',linestyle='',marker='.',markersize=8,color='k')
            axs[1].set_ylabel("Scale Factor")
            axs[1].set_ylim(0,2)
            axs[1].set_xlabel(axs[0].get_xlabel()+"[GeV]")
            axs[0].set_xlabel("")
            axs[0].legend()
            axs[1].legend()
            hep.plot.yscale_legend(ax=axs[0])
            fig.tight_layout()
            fig.savefig(Path(f"/srv/postprocessing/trigger_eff/plots/{i}/{j}/sf_plot.pdf"))
            if not(os.path.exists(path)):
                # create the directory you want to save to
                os.makedirs(path)
            with open(f"/srv/analyzer_resources/datavmc_sf/{i}/{j}_sf.pkl", 'wb') as file:
                pickle.dump(num_dict,file)
                pickle.dump(den_dict,file)
                pickle.dump(sf_h,file)
    return


def calculate_scale_factors2d():
    for i in ["2016_preVFP","2016_preVFP","2016_postVFP","2017","2018","2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"]:
        ratio_path =  list(Path(f"/srv/postprocessing/trigger_eff_2d_hist_test/plots/{i}/SingleJetEff").rglob("*.pkl"))
        with open(ratio_path[0], 'rb') as picklefile:
            num_h = pickle.load(picklefile)
        with open(ratio_path[1], 'rb') as picklefile:
            den_h = pickle.load(picklefile)
        sf_h = num_h/den_h
        plt.style.use(hep.style.CMS)
        fig, ax = plt.subplots()
        import re
        year_num = int(re.findall(r'\d+', i)[0])
        if year_num > 2018:
            energy = 13.6
        else:
            energy = 13
        hep.cms.label(year=i,ax=ax,data=True,label="Preliminary",com=energy)
        sf_h.plot2d()
        fig.tight_layout()
        fig.savefig(f"/srv/postprocessing/trigger_eff_2d_hist_test/plots/{i}/SingleJetEff/2d_sf_plot.pdf")
    return

def calculate_scale_factors3d():
    data_list = ["2016_preVFP","2016_preVFP","2016_postVFP","2017","2018","2022_preEE","2022_postEE","2023_preBPix","2023_postBPix"]
    plt.style.use(hep.style.CMS)
    for i in data_list:
        path = Path(f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/{i}/SingleJetEff").resolve()
        if not path.is_dir():
            raise ValueError(f'{path} is not a dir')
        ratio_glob =  path.glob("**/*.pkl")
        for j in ratio_glob:
            with open(j, 'rb') as picklefile:
                if 'data' in picklefile.name:
                    num_dict = pickle.load(picklefile)
                else:
                    den_dict = pickle.load(picklefile)

        num_h, num_unc, X, n = num_dict["Hist"], num_dict["Unc"], num_dict["num"], num_dict["den"]
        den_h, den_unc, Y, m = den_dict["Hist"], den_dict["Unc"], num_dict["num"], num_dict["den"]

        sf_h = num_h/den_h 
        #saving scale factor histogram
        
        sf_path = Path(f"/srv/analyzer_resources/datavmc_sf/{i}_sf.pkl")
        if not(os.path.exists(sf_path.parents[0])):
            # create the directory you want to save to
            os.makedirs(sf_path.parents[0])
        with open(sf_path.resolve(), 'wb') as sf_file:
            pickle.dump(sf_h,sf_file)

        #ones_hist = hist.Hist(*sf_h.axes)
        #sf_var = hist.Hist(*sf_h.axes)

        #ones_hist[...] = np.ones_like(sf_h.values())
        #one_hist = ones_hist[0,...]
        length = len(sf_h.axes[0])
        year_num = int(re.findall(r'\d+', i)[0])
        
        #sf_var[...] = sf_var_np

        if year_num > 2018:
            energy = 13.6
        else:
            energy = 13
        for j in range(length):
            htbin = sf_h.axes[0][j]
            htbinstr = "_".join([str(k) for k in htbin])
            fig, ax = plt.subplots()

            hep.cms.label(year=i,ax=ax,data=True,label="Preliminary",com=energy)
            sf_h[j,:,:].plot2d(ax=ax)
            fig.tight_layout()
            fig.savefig(f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/{i}/SingleJetEff/3d_sf_plot_ht_{htbinstr}.pdf")
    return

if __name__ == "__main__":
    calculate_scale_factors3d()
