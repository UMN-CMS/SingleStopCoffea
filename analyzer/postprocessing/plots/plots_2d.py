import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import hist
import hist.intervals
import mplhep
import os
import pickle


from analyzer.postprocessing.style import Styler

from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .utils import saveFig, fixBadLabels
from .mplstyles import loadStyles
from hist import Hist

def plot2D(
    packaged_hist,
    output_path,
    style_set,
    normalize=False,
    plot_configuration=None,
    color_scale="linear",
):
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    loadStyles()
    fig, ax = plt.subplots(layout="constrained")
    styler.getStyle(packaged_hist.sector_parameters)
    h = packaged_hist.histogram
    fixBadLabels(h)
    print(h)
    if normalize:
        h = h / np.sum(h.values())
    if color_scale == "log":
        art = h.plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        art = h.plot2d(ax=ax)
    labelAxis(ax, "y", h.axes)
    labelAxis(ax, "x", h.axes)
    sp = packaged_hist.sector_parameters
    addCMSBits(
        ax,
        [sp],
        extra_text=f"{sp.region_name}\n{packaged_hist.title}",
        text_color="white",
        plot_configuration=pc,
    )
    saveFig(fig, output_path, extension=pc.image_type)
    plt.close(fig)


def getContour(HH, val):
    total = np.sum(HH)
    for i in range(round(np.max(HH))):
        if np.sum(HH[HH > i]) < (total * val):
            return i
    return None

def plot2DSigBkg(
    bkg_hist,
    sig_hist,
    output_path,
    style_set,
    normalize=False,
    plot_configuration=None,
    color_scale="linear",
    override_axis_labels=None,
):
    override_axis_labels = override_axis_labels or {}
    pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    loadStyles()
    fig, ax = plt.subplots(layout="constrained")
    styler.getStyle(bkg_hist.sector_parameters)
    h = bkg_hist.histogram
    fixBadLabels(h)

    if normalize:
        h = h / np.sum(h.values())
    if color_scale == "log":
        art = h.plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        art = h.plot2d(ax=ax)

    from scipy.ndimage import gaussian_filter

    sh = sig_hist.histogram

    HH, xe, ye = sh.to_numpy()
    HH = gaussian_filter(HH, 1.2)
    midpoints = (xe[1:] + xe[:-1]) / 2, (ye[1:] + ye[:-1]) / 2
    grid = HH.transpose()
    h.sum().value

    sig_style = sig_hist.style or styler.getStyle(sig_hist.sector_parameters)

    ax.contour(
        *midpoints,
        grid,
        [getContour(HH, x) for x in (0.75, 0.5, 0.25)],
        linewidths=sig_style.line_width,
        colors=[sig_style.color],
    )

    labelAxis(ax, "y", h.axes, label=override_axis_labels.get("y"))
    labelAxis(ax, "x", h.axes, label=override_axis_labels.get("x"))

    proxy = [
        plt.Line2D(
            [0],
            [0],
            lw=sig_style.line_width or 2,
            color=sig_style.color,
            label=sig_hist.title,
        )
    ]

    sp = bkg_hist.sector_parameters
    ax.legend(
        handles=proxy,
        facecolor=pc.legend_fill_color,
        framealpha=pc.legend_fill_alpha,
        frameon=True,
    )

    addCMSBits(
        ax,
        [sp],
        extra_text=f"{sp.region_name}\n{bkg_hist.title}",
        text_color="white",
        plot_configuration=plot_configuration,
    )
    saveFig(fig, output_path, extension=plot_configuration.image_type)
    plt.close(fig)

def plotRatio2D(
    denominator,
    numerators,
    output_path,
    style_set,
    normalize=False,
    color_scale="linear",
    plot_configuration=None,
):
    #pc = plot_configuration or PlotConfiguration()
    styler = Styler(style_set)
    matplotlib.use("Agg")
    loadStyles()
    fig, ax = plt.subplots()

    den_hist = denominator.histogram

    fixBadLabels(den_hist)

    #style = denominator.style or styler.getStyle(denominator.sector_parameters)

    all_ratios, all_uncertainties = [], []

    for num in numerators:
        num_hist = num.histogram

        fixBadLabels(num_hist)

        s = num.style or styler.getStyle(num.sector_parameters)
 
        # r1 = np.concatenate((2*np.ones(23,dtype=int),3*np.ones(18,dtype=int)))
        # r2 = np.concatenate((2*np.ones(3,dtype=int),3*np.ones(8,dtype=int)))

        # r1 = np.concatenate((2*np.ones(22,dtype=int),4*np.ones(14,dtype=int)))
        # r2 = np.concatenate((2*np.ones(3,dtype=int),4*np.ones(6,dtype=int)))

        r1 = np.concatenate((np.ones(1,dtype=int),2*np.ones(22,dtype=int),11*np.ones(5,dtype=int)))
        r2 = np.concatenate((np.ones(1,dtype=int),2*np.ones(2,dtype=int),5*np.ones(5,dtype=int)))
        rebin_pt = hist.rebin(groups=r1)
        rebin_sd = hist.rebin(groups=r2)

        temp_num_name = num_hist.axes.name
        temp_den_name = den_hist.axes.name
        temp_num_unit = num_hist.axes.unit
        temp_den_unit = den_hist.axes.unit

        #this requires boost_histogram >= 1.5.1
        num_hist = num_hist[::rebin_pt,::rebin_sd]
        den_hist = den_hist[::rebin_pt,::rebin_sd]

        #shouldn't be needed after boost_histogram 1.5.2
        num_hist.axes.name = temp_num_name
        den_hist.axes.name = temp_den_name
        num_hist.axes.unit = temp_num_unit
        den_hist.axes.unit = temp_den_unit

        ratio_hist = hist.Hist(*num_hist.axes)
        ratio_unc_hist_lower = hist.Hist(*num_hist.axes)
        ratio_unc_hist_upper = hist.Hist(*num_hist.axes)

        n = num_hist.values()
        d = den_hist.values()
        
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_unc = hist.intervals.ratio_uncertainty(n,d,uncertainty_type="efficiency")
            if normalize:
                rvalues = (n / np.sum(n)) / (d / np.sum(d))
                ratio_unc *= (np.sum(d)/np.sum(n))
            else:
                rvalues = (n / d)
        
        ratio_hist[...] = rvalues
        ratio_unc_hist_lower[...] = ratio_unc[0]
        ratio_unc_hist_upper[...] = ratio_unc[1]
        all_ratios.append(ratio_hist)

        if color_scale == "log":
            art = ratio_hist.plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
        else:
            art = ratio_hist.plot2d(ax=ax)
        
        ax.vlines(450,*ax.get_ylim(),colors='r')
        ax.hlines(50,*ax.get_xlim(),colors='r')
        #Saving Ratio
        ax = art.pcolormesh.axes
        fig = ax.get_figure()
        labelAxis(ax, "y", ratio_hist.axes)
        labelAxis(ax, "x", ratio_hist.axes)
        addCMSBits(
            ax,
            [denominator.sector_parameters, num.sector_parameters],
            extra_text=f"{num.sector_parameters.region_name}\n{num.sector_parameters.dataset.title}",
            plot_configuration=plot_configuration,
        )
        mplhep.sort_legend(ax=ax)
        fig.tight_layout()
        #Saving Ratio
        import os
        head, tail = os.path.split(output_path)
        os.makedirs(head, exist_ok=True)
        import pickle
        with open(output_path+f"_ratio2d_hist.pkl", 'wb') as file:
            pickle.dump(ratio_hist, file)
        saveFig(fig, output_path, extension=plot_configuration.image_type)
        plt.close(fig)
        
    
    saveFig(fig, output_path, extension=plot_configuration.image_type)

def plot3D(
    histogram,
    output_path,
    style_set,
    normalize=False,
    color_scale="linear",
    plot_configuration=None,
    save_plots=True,
):
    hist_3d = histogram.histogram
    fixBadLabels(hist_3d)

    head, _tail = os.path.split(output_path) 
    os.makedirs(head, exist_ok=True)
    r0 = np.concatenate((28*np.ones(1,dtype=int),4*np.ones(8,dtype=int),60*np.ones(1,dtype=int)))
    r1 = 25*np.ones(4,dtype=int)
    #r2 = np.concatenate((np.ones(1,dtype=int),2*np.ones(2,dtype=int),5*np.ones(5,dtype=int)))
    r2 = 6*np.ones(5,dtype=int)

    rebin_ht = hist.rebin(groups=r0)
    rebin_pt = hist.rebin(groups=r1)
    rebin_sd = hist.rebin(groups=r2)

    temp_name = hist_3d.axes.name
    temp_unit = hist_3d.axes.unit

    #this requires boost_histogram >= 1.5.1
    hist_3d = hist_3d[::rebin_ht, ::rebin_pt, ::rebin_sd]

    #shouldn't be needed after boost_histogram 1.5.2
    hist_3d.axes.name = temp_name
    hist_3d.axes.unit = temp_unit

    length = len(hist_3d.axes[0])
    for i in range(length):
        htbin = hist_3d.axes[0][i]
        htbinstr = "_".join([str(k) for k in htbin])

        #Plotting
        if save_plots: 
            slice_plot_2d(hist=hist_3d, 
                bin_index=i, 
                plot_config=plot_configuration, 
                text=None, 
                output_path=output_path, 
                sec_params=histogram.sector_parameters,
                htbinstr=htbinstr,
                style_set=style_set,
                color_scale=color_scale)

    with open(output_path+".pkl", 'wb') as file:
        pickle.dump(hist_3d, file)


def plotRatio3D(
    denominator,
    numerators,
    output_path,
    style_set,
    normalize=False,
    color_scale="linear",
    plot_configuration=None,
):
    den_hist = denominator.histogram
    den = denominator
    fixBadLabels(den_hist)

    #style = denominator.style or styler.getStyle(denominator.sector_parameters)

    all_ratios, all_uncertainties = [], []

    for num in numerators:
        num_hist = num.histogram
        fixBadLabels(num_hist)

        r0 = np.concatenate((28*np.ones(1,dtype=int),4*np.ones(8,dtype=int),60*np.ones(1,dtype=int)))
        r1 = 25*np.ones(4,dtype=int)
        #r2 = np.concatenate((np.ones(1,dtype=int),2*np.ones(2,dtype=int),5*np.ones(5,dtype=int)))
        r2 = 6*np.ones(5,dtype=int)

        rebin_ht = hist.rebin(groups=r0)
        rebin_pt = hist.rebin(groups=r1)
        rebin_sd = hist.rebin(groups=r2)

        temp_num_name = num_hist.axes.name
        temp_den_name = den_hist.axes.name
        temp_num_unit = num_hist.axes.unit
        temp_den_unit = den_hist.axes.unit

        #this requires boost_histogram >= 1.5.1
        num_hist = num_hist[::rebin_ht, ::rebin_pt, ::rebin_sd]
        den_hist = den_hist[::rebin_ht, ::rebin_pt, ::rebin_sd]

        #shouldn't be needed after boost_histogram 1.5.2
        num_hist.axes.name = temp_num_name
        den_hist.axes.name = temp_den_name
        num_hist.axes.unit = temp_num_unit
        den_hist.axes.unit = temp_den_unit

        ratio_hist = hist.Hist(*num_hist.axes)
        ratio_unc_hist_lower = hist.Hist(*num_hist.axes)
        ratio_unc_hist_upper = hist.Hist(*num_hist.axes)

        n = num_hist.values()
        d = den_hist.values()
        with np.errstate(divide="ignore", invalid="ignore"):
            ratio_unc = hist.intervals.ratio_uncertainty(n,d,uncertainty_type="efficiency")
            if normalize:
                rvalues = (n / np.sum(n)) / (d / np.sum(d))
                ratio_unc *= (np.sum(d)/np.sum(n))
            else:
                rvalues = (n / d)
        rvalues[n==0]=0
        ratio_unc[0][n==0]=0
        ratio_unc[1][n==0]=0
        ratio_hist[...] = rvalues
        ratio_unc_hist_lower[...] = ratio_unc[0]
        ratio_unc_hist_upper[...] = ratio_unc[1]
        all_ratios.append(ratio_hist)
        length = len(ratio_hist.axes[0])
        
        head, tail = os.path.split(output_path) 
        os.makedirs(head, exist_ok=True)

        for i in range(length):
            htbin = ratio_hist.axes[0][i]
            htbinstr = "_".join([str(k) for k in htbin])

            #Plotting and saving efficiency 
            slice_ratio_plot_2d(hist=ratio_hist, 
                bin_index=i,
                den_sec_params=den.sector_paramters,
                num_sec_params=num.sector_parameters,
                plot_config=plot_configuration, 
                text=None, 
                output_path=output_path, 
                htbinstr=htbinstr,
                style_set=style_set,
                color_scale=color_scale)
        rh_with_unc = {"Hist": ratio_hist, "Unc": ratio_unc, "num": num_hist, "den": den_hist}
        with open(output_path+".pkl", 'wb') as file:
            pickle.dump(rh_with_unc, file)

def slice_plot_2d(hist, bin_index, sec_params, plot_config, text, output_path, htbinstr, style_set, color_scale):
    #Plotting and saving lower uncertainty
    styler = Styler(style_set)
    matplotlib.use("Agg")
    loadStyles()
    fig, ax = plt.subplots()
    offset = matplotlib.colors.TwoSlopeNorm(vcenter=1)
    if color_scale == "log":
        art = hist[bin_index,:,:].plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        art = hist[bin_index,:,:].plot2d(ax=ax)

    ax = art.pcolormesh.axes
    fig = ax.get_figure()

    labelAxis(ax, "y", hist[bin_index,:,:].axes)
    labelAxis(ax, "x", hist[bin_index,:,:].axes)
    if text is not None:
        extra_text = f"{text}\n{sec_params.region_name}\n{sec_params.dataset.title}\nHT: {htbinstr} GeV"
    else:
        extra_text = f"{sec_params.region_name}\n{sec_params.dataset.title}\nHT: {htbinstr} GeV"
    addCMSBits(
        ax,
        [sec_params],
        extra_text=extra_text,
        plot_configuration=plot_config,
        )
    mplhep.sort_legend(ax=ax)
    fig.tight_layout()
    if text is None:
        saveFig(fig, output_path+htbinstr, extension=plot_config.image_type)
    else:
        saveFig(fig, output_path+f"_{text}"+htbinstr, extension=plot_config.image_type)
    plt.close(fig)

def slice_ratio_plot_2d(hist, bin_index, den_sec_params, num_sec_params, plot_config, text, output_path, htbinstr, style_set, color_scale):
    #Plotting and saving lower uncertainty
    styler = Styler(style_set)
    matplotlib.use("Agg")
    loadStyles()
    fig, ax = plt.subplots()
    offset = matplotlib.colors.TwoSlopeNorm(vcenter=1)
    if color_scale == "log":
        art = hist[bin_index,:,:].plot2d(norm=matplotlib.colors.LogNorm(), ax=ax)
    else:
        art = hist[bin_index,:,:].plot2d(ax=ax)

    #ax.vlines(500,*ax.get_ylim(),colors='r')
    #ax.hlines(50,*ax.get_xlim(),colors='r')

    ax = art.pcolormesh.axes
    fig = ax.get_figure()

    labelAxis(ax, "y", hist[bin_index,:,:].axes)
    labelAxis(ax, "x", hist[bin_index,:,:].axes)
    if text is not None:
        extra_text = f"{text}\n{num_sec_params.region_name}\n{num_sec_params.dataset.title}\nHT: {htbinstr} GeV"
    else:
        extra_text = f"{num_sec_params.region_name}\n{num_sec_params.dataset.title}\nHT: {htbinstr} GeV"
    addCMSBits(
        ax,
        [den_sec_params, num_sec_params],
        extra_text=extra_text,
        plot_configuration=plot_config,
        )
    mplhep.sort_legend(ax=ax)
    fig.tight_layout()
    if text is None:
        saveFig(fig, output_path+htbinstr, extension=plot_config.image_type)
    else:
        saveFig(fig, output_path+f"_{text}"+htbinstr, extension=plot_config.image_type)
    plt.close(fig)


