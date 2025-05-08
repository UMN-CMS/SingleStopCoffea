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
    #data_list = ["2018"]
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
        den_h, den_unc, Y, m = den_dict["Hist"], den_dict["Unc"], den_dict["num"], den_dict["den"]
        
        sf_h = num_h/den_h
        sf_h_values = sf_h.values()
        sf_h_values[num_h.values()==0] = 0
        sf_h[...] = sf_h_values
        #uncertainty
        #X/n and Y/m are Binomial and (X/n)/(Y/m) is approximately log normal with the following variance
        #var = 1/X+1/Y-1/m-1/n
        log_sf_var = np.divide(1,X.values())+np.divide(1,Y.values())-np.divide(1,m.values())-np.divide(1,n.values())
        log_upper = np.log(sf_h.values()) + np.sqrt(log_sf_var)
        log_lower = np.log(sf_h.values()) - np.sqrt(log_sf_var)

        upper = np.exp(log_upper)
        lower = np.exp(log_lower)
        
        unc_upper = upper-sf_h.values()
        unc_lower = sf_h.values()-lower

        sf_unc_hist_lower = hist.Hist(*sf_h.axes)
        sf_unc_hist_upper = hist.Hist(*sf_h.axes)
        
        sf_unc_hist_lower[...] = unc_lower
        sf_unc_hist_upper[...] = unc_upper

        sf_path = Path(f"/srv/analyzer_resources/datavmc_sf/{i}_sf.pkl")
        if not(os.path.exists(sf_path.parents[0])):
            # create the directory you want to save to
            os.makedirs(sf_path.parents[0])
        with open(sf_path.resolve(), 'wb') as sf_file:
            pickle.dump(sf_h,sf_file)
        save_plots(sf_h, i, sf_unc_hist_lower, sf_unc_hist_upper, loop_axis = 1)

def save_plots(sf_h, dataset, unc_lower, unc_upper, loop_axis = 0):
    length = len(sf_h.axes[loop_axis])
    year_num = int(re.findall(r'\d+', dataset)[0])
        
    plt.style.use(hep.style.CMS)
    if year_num > 2018:
        energy = 13.6
    else:
        energy = 13
    for j in range(length):
        loop_bin = sf_h.axes[loop_axis][j]
        loop_bin_str = "_".join([str(k) for k in loop_bin])
        fig, ax = plt.subplots()
        if loop_axis == 0:
            x_centers = sf_h.axes[1].centers
            label = f'HT {loop_bin_str}\nScale Factors' 
        elif loop_axis == 1:
            x_centers = sf_h.axes[0].centers
            label = f'PT {loop_bin_str}\nScale Factors' 

        msd_centers = sf_h.axes[2].centers
        offset = matplotlib.colors.TwoSlopeNorm(vmin=0, vcenter=1)
        hep.cms.label(year=dataset,ax=ax,data=True,label=label,com=energy,loc=3)
        if loop_axis == 0:
            sf_h[j,:,:].plot2d(norm=offset, ax=ax, cmap='berlin')

            for x_bin in range(len(x_centers)):
                for msd_bin in range(len(msd_centers)):
                    # Auto text color based on background
                    if np.isfinite(sf_h[j,x_bin, msd_bin]):
                        text_color = 'white'
                    else:
                        text_color = 'black'
                        
                    sup = f'{unc_upper[j,x_bin,msd_bin]:.2f}'
                    sub = f'{unc_lower[j,x_bin,msd_bin]:.2f}'
                    ax.text(x_centers[x_bin], msd_centers[msd_bin], f"${sf_h[j,x_bin,msd_bin]:.2f}^{{{sup}}}_{{{sub}}}$",
                        ha="center", va="center",
                        color=text_color, fontsize=12) 

        elif loop_axis == 1:
            sf_h[:,j,:].plot2d(norm=offset, ax=ax, cmap='berlin')

            for x_bin in range(len(x_centers)):
                for msd_bin in range(len(msd_centers)):
                    # Auto text color based on background
                    if np.isfinite(sf_h[x_bin,j, msd_bin]):
                        text_color = 'white'
                    else:
                        text_color = 'black'
                        
                    sup = f'{unc_upper[x_bin,j,msd_bin]:.2f}'
                    sub = f'{unc_lower[x_bin,j,msd_bin]:.2f}'
                    ax.text(x_centers[x_bin], msd_centers[msd_bin], f"${sf_h[x_bin,j,msd_bin]:.2f}^{{{sup}}}_{{{sub}}}$",
                        ha="center", va="center",
                        color=text_color, fontsize=12) 
        #hep.cms.text(text=f'HT {htbinstr}\nScale Factors', ax=ax, loc=3, com=energy, year=dataset, data=True)
        fig.tight_layout()
        if loop_axis == 0:
            output_title = f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/{dataset}/SingleJetEff/3d_sf_plot_ht_{loop_bin_str}.pdf"
        #    lower_max = np.nanmax(unc_lower[j,:,:].values())
        #    upper_max = np.nanmax(unc_upper[j,:,:].values())

        elif loop_axis == 1:
            output_title = f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/{dataset}/SingleJetEff/3d_sf_plot_pt_{loop_bin_str}.pdf"
        #    lower_max = np.nanmax(unc_lower[:,j,:].values())
        #    upper_max = np.nanmax(unc_upper[:,j,:].values())

        fig.savefig(output_title)
        plt.close()
        #vmax = np.max([upper_max, lower_max])
        #save_unc(unc_lower, 'lower', dataset, loop_bin_str, j, energy, vmax, loop_axis)
        #save_unc(unc_upper, 'upper', dataset, loop_bin_str, j, energy, vmax, loop_axis)
       
def save_unc(unc, title, dataset, loop_bin_str, index, energy, vmax, loop_axis):
        fig, ax = plt.subplots()
        
        norm = matplotlib.colors.Normalize(vmin=0, vmax=vmax)
        if loop_axis == 0:
            x_centers = unc.axes[1].centers
            hep.cms.label(year=dataset,ax=ax,data=True,label=f'HT: {loop_bin_str}\nSF {title} unc',com=energy,loc=3)
            unc[index,:,:].plot2d(ax=ax, norm=norm)

        elif loop_axis == 1:
            x_centers = unc.axes[0].centers
            hep.cms.label(year=dataset,ax=ax,data=True,label=f'PT: {loop_bin_str}\nSF {title} unc',com=energy,loc=3)
            unc[:,index,:].plot2d(ax=ax, norm=norm)

        msd_centers = unc.axes[2].centers
        if loop_axis == 0:
            output_title = f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/{dataset}/SingleJetEff/3d_sf_plot_ht_{loop_bin_str}_unc_{title}.pdf"
            for x_bin in range(len(x_centers)):
                for msd_bin in range(len(msd_centers)):
                    # Auto text color based on background
                    if np.isfinite(unc[index, x_bin, msd_bin]):
                        text_color = 'white'
                    else:
                        text_color = 'black'
                        
                    ax.text(x_centers[x_bin], msd_centers[msd_bin], f"{unc[index,x_bin,msd_bin]:.2f}",
                        ha="center", va="center",
                        color=text_color, fontsize=12) 
        elif loop_axis == 1:
            output_title = f"/srv/postprocessing/trigger_eff_3d_hist_test/plots/{dataset}/SingleJetEff/3d_sf_plot_pt_{loop_bin_str}_unc_{title}.pdf"
            for x_bin in range(len(x_centers)):
                for msd_bin in range(len(msd_centers)):
                    # Auto text color based on background
                    if np.isfinite(unc[x_bin, index, msd_bin]):
                        text_color = 'white'
                    else:
                        text_color = 'black'
                        
                    ax.text(x_centers[x_bin], msd_centers[msd_bin], f"{unc[x_bin,index,msd_bin]:.2f}",
                        ha="center", va="center",
                        color=text_color, fontsize=12) 

        fig.tight_layout()
        fig.savefig(output_title)
        plt.close()


if __name__ == "__main__":
    calculate_scale_factors3d()
