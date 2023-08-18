import pickle as pkl
import sys
sys.path.append(".")
import hist
import matplotlib.pyplot as plt
from analyzer.plotting.core_plots import *
from analyzer.run_analysis import runModulesOnSamples
from analyzer.core import modules as all_modules
from pathlib import Path

loadStyles()


backgrounds=["Skim_QCDInclusive2018"]
compressed = [f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400")]
uncompressed = [f"signal_312_{p}" for p in ("2000_1400", "1200_400", "1500_900")]
both = compressed + uncompressed


file_name="all_hists.pkl"
data = pkl.load(open(file_name, 'rb'))
histos = data["histograms"]

def simplePlot(hist, sig_set, scale="log", title=""):
    h = histos[hist]
    with open(savedir / "descriptions.txt", "a") as f:
        f.write(f"{hist}: {h.description}\n")
    hc = h[{"dataset" : backgrounds + sig_set}]
    if len(h.axes) == 2:
        fig,ax = drawAs1DHist(hc,cat_axis="dataset", cat_filter="^(?!signal)", yerr=True)
        drawAsScatter(ax, hc,cat_axis="dataset", cat_filter="signal", yerr=True)
        addTitles1D(ax,hc)
        addPrelim(ax)
        ax.set_yscale(scale)
        name = h.name
        fig.savefig(savedir / f"{hist}.pdf")
        plt.close(fig)
    elif len(h.axes) == 3:
        for x in hc.axes[0]:
            realh = hc[{"dataset" : x}]
            fig,ax = drawAs2DHist(PlotObject.create(realh))
            addTitles2D(ax,realh)
            addPrelim(ax)
            name = h.name
            fig.savefig(savedir / f"{hist}_{x}.pdf")
            plt.close(fig)

        
    

savedir = Path("figures")
savedir.mkdir(exist_ok=True, parents=True)
open(savedir / "descriptions.txt", "wt")


print(histos.keys())
simplePlot("HT", compressed)
simplePlot("m14_m", both)
simplePlot("m13_m", compressed)
simplePlot("m24_m", uncompressed)

simplePlot("m3_top_3_no_lead_b", uncompressed)
simplePlot("m3_top_2_plus_lead_b", compressed)

simplePlot("m14_vs_m3_top_3_no_lead_b", uncompressed)
simplePlot("m14_vs_m3_top_2_plus_lead_b", compressed)

simplePlot("ratio_m14_vs_m3_top_3_no_lead_b", uncompressed)
simplePlot("ratio_m14_vs_m3_top_2_plus_lead_b", compressed)

simplePlot("lead_medium_bjet_ordinality", both)
simplePlot("sublead_medium_bjet_ordinality", both)

simplePlot("m14_vs_m24", uncompressed)
simplePlot("m14_vs_m13", compressed)
simplePlot("ratio_m14_vs_m24", uncompressed)
simplePlot("ratio_m14_vs_m13", compressed)
