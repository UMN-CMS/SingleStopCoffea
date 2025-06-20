import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from analyzer.postprocessing.style import Styler
import hist
from .annotations import addCMSBits, labelAxis
from .common import PlotConfiguration
from .utils import saveFig, fixBadLabels
from .mplstyles import loadStyles


def plot2D(
    packaged_hist,
    group_params,
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
    fig, ax = plt.subplots()
    style = styler.getStyle(packaged_hist.sector_parameters)
    h = packaged_hist.histogram
    fixBadLabels(h)

    #r0 = 5*np.ones(25,dtype=int)
    #r1 = 5*np.ones(25,dtype=int)

    #rebin_mstop = hist.rebin(groups=r0)
    #rebin_mratio = hist.rebin(groups=r1)
    temp_name = h.axes.name
    temp_unit = h.axes.unit 
    h = h[hist.rebin(2),hist.rebin(2)]
    h.axes.name = temp_name
    h.axes.unit = temp_unit
    
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
        extra_text=f"{sp.region_name}\n{sp.dataset.title}",
        text_color="white",
        plot_configuration=plot_configuration,
    )
    saveFig(fig, output_path, extension=plot_configuration.image_type)
    plt.close(fig)
