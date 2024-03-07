import pickle as pkl
import sys

sys.path.append(".")
import hist as hst
import matplotlib.pyplot as plt

from analyzer.plotting.styles import *
from analyzer.plotting.core_plots import *
from analyzer.plotting.simple_plot import *
from analyzer.datasets import loadSamplesFromDirectory

from pathlib import Path

import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

loadStyles()


backgrounds = ["QCDInclusive2018", "TTToHadronic2018"]
compressed = [f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400")]
uncompressed = [f"signal_312_{p}" for p in ("2000_1400", "1200_400", "1500_900")]
both = compressed + uncompressed
representative = [f"signal_312_{p}" for p in ("2000_1900", "1200_400", "1500_900")]

manager = loadSamplesFromDirectory("datasets")

file_name = "singlemuon_softdrop.pkl"
data = pkl.load(open(file_name, "rb"))
histos = data["histograms"]
lumi = data["target_lumi"]

def simplePlot(
    hist,
    sig_set,
    bkg_set,
    scale="log",
    title="",
    add_name="",
    normalize=False,
    sig_style="hist",
    add_label=None,
    selection = {},
):
    if add_label is None and add_name:
        add_label = add_name.title()
    print(f"Now plotting {hist}")
    add_name = add_name + "_" if add_name else ""

    if hist == 'nJets':    
      selection += { 'jetpT300': sum, 'nJets456': sum, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': sum, '313Bs': sum, 'dRbb_312': sum, 'dRbb_313': sum }
    elif (hist == 'pT1') and ('DataSingleMuon2018' not in sig_set):
      selection += { 'jetpT300': sum, 'nJets456': 1, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': 1, '313Bs': sum, 'dRbb_312': 1, 'dRbb_313': sum }
    elif hist == 'dRbb12':
      selection += { 'jetpT300': 1, 'nJets456': 1, 'leptonVeto': 1, 'dRJets24': sum, '312Bs': 1, '313Bs': sum, 'dRbb_312': sum, 'dRbb_313': sum }

    h = histos[hist][selection]
    with open(savedir / "descriptions.txt", "a") as f:
        f.write(f"{hist}: {h.description}\n")
    datasets = sorted(bkg_set) + sorted(sig_set)
    hc = h[{"dataset": datasets}]
    if normalize:
        hc = getNormalized(hc, "dataset")
    #if hist == 'pT1': hc[::2j]
    if len(h.axes) == 2:
        fig, ax = drawAs1DHist(
            hc,
            plot_name = hist,
            cat_axis="dataset",
            manager=manager,
            cat_filter="^(?!signal)",
            yerr=False,
        )
        if sig_style == "scatter":
            drawAsScatter(
                ax,
                hc,
                cat_axis="dataset",
                cat_filter="signal",
                manager=manager,
                yerr=True,
            )
        elif sig_style == "hist":
            drawAs1DHist(
                ax,
                hc,
                cat_axis="dataset",
                cat_filter="signal",
                manager=manager,
                yerr=True,
                fill=False,
            )

        ax.set_yscale(scale)
        addEra(ax, lumi or 59.8)
        addPrelim(ax, additional_text="\n$\\lambda_{312}''$" + " Selection" + (add_label or ""))

        addTitles1D(ax, hc, top_pad=0.4)
        handles, labels = ax.get_legend_handles_labels()
        print(handles, labels)
        labels, handles = zip(
          *reversed(sorted(zip(labels, handles), key=lambda t: t[0]))
        )
        ax.legend(handles, labels)

        fig.tight_layout()
        fig.savefig(savedir / f"{add_name}{hist}.pdf")
        plt.close(fig)
    elif len(h.axes) == 3:
        for x in hc.axes[0]:
            realh = hc[{"dataset": x}]
            if sig_style == "hist":
                fig, ax = drawAs2DHist(PlotObject(realh, x, manager[x]))
                addTitles2D(ax, realh)
                addPrelim(ax, "out")
                addEra(ax, lumi or 59.8)
            elif sig_style == "profile":
                fig, ax = drawAs2DExtended(
                    PlotObject(realh, x, manager[x]),
                    top_stack=[PlotObject(realh[sum, :], x, manager[x])],
                    right_stack=[PlotObject(realh[:, sum], x, manager[x])],
                )
                addTitles2D(ax, realh)
                addPrelim(
                    ax,
                    additional_text="\n$\\lambda_{312}''$" + f" {add_label}," + f"{manager[x].getTitle()}",
                    pos="in",
                    color="white",
                )
                addEra(ax.top_axes[-1], lumi or 59.8)
            name = h.name
            fig.savefig(savedir / f"{add_name}{hist}_{x}.pdf")
            plt.close(fig)


savedir = Path("figures")
savedir.mkdir(exist_ok=True, parents=True)
open(savedir / "descriptions.txt", "wt")

singlemuon = ["DataSingleMuon2018"]
#simplePlot('pT1', representative, backgrounds, scale = 'linear', normalize=True)
#simplePlot('dRbb12', representative, backgrounds, scale = 'linear', normalize = True)
#simplePlot('nJets', representative, backgrounds, scale = 'linear', normalize = True)
simplePlot('mSoftDrop2D', singlemuon, [], scale = 'linear', normalize=False, selection = {'pT400': sum, 'HT1050': sum})

'''
pT_pass = histos['pT1'][{'pT400': 1, 'HT1050': sum}][{'dataset': 'DataSingleMuon2018'}]
pT_total = histos['pT1'][{'pT400': sum, 'HT1050': sum}][{'dataset': 'DataSingleMuon2018'}]
HT_pass = histos['pT1'][{'pT400': sum, 'HT1050': 1}][{'dataset': 'DataSingleMuon2018'}]
HT_total = histos['pT1'][{'pT400': sum, 'HT1050': sum}][{'dataset': 'DataSingleMuon2018'}]

ptp_PO = PlotObject(pT_pass, r'pT_1', None)
ptt_PO = PlotObject(pT_total, r'pT_1', None)
htp_PO = PlotObject(HT_pass, r'HT', None)
htt_PO = PlotObject(HT_total, r'HT', None)

print(ptp_PO)

fig, ax = drawAs1DHist(ptp_PO, yerr=True, fill=False)
ax.set_yscale('log')
drawAs1DHist(ax, ptt_PO, yerr=True, fill=False)
addAxesToHist(ax, num_bottom=1, bottom_pad=0)
ab = ax.bottom_axes[0]
drawRatio(ab, ptp_PO, ptt_PO)
ab.set_ylabel("Efficiency")

addEra(ax, 59.8)

fig.tight_layout()
fig.savefig("figures/pT_Efficiency.pdf")
plt.close(fig)

fig, ax = drawAs1DHist(htp_PO, yerr=True, fill=False)
ax.set_yscale('log')
drawAs1DHist(ax, htt_PO, yerr=True, fill=False)
addAxesToHist(ax, num_bottom=1, bottom_pad=0)
ab = ax.bottom_axes[0]
drawRatio(ab, htp_PO, htt_PO)
ab.set_ylabel("Efficiency")

addEra(ax, 59.8)

fig.tight_layout()
fig.savefig("figures/HT_Efficiency.pdf")
plt.close(fig)

softDrop_pass = histos['mSoftDrop'][{'pT400': 1, 'HT1050': sum}][{'dataset': 'DataSingleMuon2018'}]
softDrop_total = histos['mSoftDrop'][{'pT400': sum, 'HT1050': sum}][{'dataset': 'DataSingleMuon2018'}]
sdp_PO = PlotObject(softDrop_pass, r'm_{Soft Drop}', None)
sdt_PO = PlotObject(softDrop_total, r'm_{Soft Drop}', None)

fig, ax = drawAs1DHist(sdp_PO, yerr=True, fill=False)
ax.set_yscale('log')
drawAs1DHist(ax, sdt_PO, yerr=True, fill=False)
addAxesToHist(ax, num_bottom=1, bottom_pad=0)
ab = ax.bottom_axes[0]
drawRatio(ab, sdp_PO, sdt_PO)
ab.set_ylabel("Efficiency")

addEra(ax, 59.8)

fig.tight_layout()
fig.savefig("figures/softDrop_Efficiency.pdf")
plt.close(fig)
'''

softDrop2D_pass = histos['mSoftDrop2D'][{'pT400': 1, 'HT1050': sum}][{'dataset': 'DataSingleMuon2018'}]
softDrop2D_total = histos['mSoftDrop2D'][{'pT400': sum, 'HT1050': sum}][{'dataset': 'DataSingleMuon2018'}]
sdp2D_PO = PlotObject(softDrop2D_pass, r'm_{Soft Drop}', None)
sdt2D_PO = PlotObject(softDrop2D_total, r'm_{Soft Drop}', None)

#fig, ax = drawAs2DHist(sdp2D_PO) #, yerr=True, fill=False)
#ax.set_yscale('log')
#drawAs2DHist(sdt2D_PO) #, yerr=True, fill=False)
#addAxesToHist(ax, num_bottom=2, bottom_pad=0)
#ab = ax.bottom_axes[0]
fig, ax = drawRatio2D(sdp2D_PO, sdt2D_PO)
ax.set_ylabel("mSoftDrop")

addEra(ax, 59.8)

fig.tight_layout()
fig.savefig("figures/softDrop2D_Efficiency.pdf")
plt.close(fig)


'''
simplePlot("m14_vs_m13", compressed, sig_style="profile", add_label="Compressed")
sys.exit(0)
simplePlot("h_njet", compressed, add_name="compressed")
simplePlot("h_njet", uncompressed, add_name="uncompressed")

simplePlot("m14_vs_m3_top_3_no_lead_b", uncompressed)
simplePlot("m14_vs_m3_top_2_plus_lead_b", compressed)

simplePlot("medium_bdr", compressed, add_name="compressed")
simplePlot("medium_bdr", uncompressed, add_name="uncompressed")

simplePlot("medium_bb_eta", compressed, add_name="compressed")
simplePlot("medium_bb_eta", uncompressed, add_name="uncompressed")

simplePlot("medium_bb_phi", compressed, add_name="compressed")
simplePlot("medium_bb_phi", uncompressed, add_name="uncompressed")




simplePlot(
    "chi_b_dr",
    compressed,
    [],
    add_name="compressed",
    normalize=True,
    scale="linear",
    sig_style="hist",
)
simplePlot(
    "chi_b_eta",
    compressed,
    [],
    add_name="compressed",
    normalize=True,
    scale="linear",
    sig_style="hist",
)
simplePlot(
    "chi_b_phi",
    compressed,
    [],
    add_name="compressed",
    normalize=True,
    scale="linear",
    sig_style="hist",
)

simplePlot(
    "chi_b_dr",
    uncompressed,
    [],
    add_name="uncompressed",
    normalize=True,
    scale="linear",
    sig_style="hist",
)
simplePlot(
    "chi_b_eta",
    uncompressed,
    [],
    add_name="uncompressed",
    normalize=True,
    scale="linear",
    sig_style="hist",
)
simplePlot(
    "chi_b_phi",
    uncompressed,
    [],
    add_name="uncompressed",
    normalize=True,
    scale="linear",
    sig_style="hist",
)

simplePlot("HT", representative)

simplePlot("m14_m", representative)
simplePlot("m13_m", compressed, add_label="Compressed")
simplePlot("m24_m", uncompressed, add_label="Uncompressed")

simplePlot(
    "m3_top_3_no_lead_b",
    uncompressed,
)
simplePlot("m3_top_2_plus_lead_b", compressed)


simplePlot("ratio_m14_vs_m3_top_3_no_lead_b", uncompressed)
simplePlot("ratio_m14_vs_m3_top_2_plus_lead_b", compressed)

simplePlot("lead_medium_bjet_ordinality", compressed, add_name="compressed")
simplePlot("lead_medium_bjet_ordinality", uncompressed, add_name="uncompressed")

simplePlot("sublead_medium_bjet_ordinality", compressed, add_name="compressed")
simplePlot("sublead_medium_bjet_ordinality", uncompressed, add_name="uncompressed")

simplePlot(
    "num_top_3_jets_matched_chi_children", compressed, [], add_label="Compressed"
)
simplePlot("num_top_4_jets_matched_stop_children", representative, [])
simplePlot(
    "num_sub_3_jets_matched_chi_children", uncompressed, [], add_label="Uncompressed"
)

simplePlot(
    "mstop_gen_matched",
    compressed + uncompressed,
    [],
    normalize=True,
    scale="linear",
)
simplePlot(
    "mchi_gen_matched",
    compressed + uncompressed,
    [],
    normalize=True,
    scale="linear",
)

simplePlot("m14_vs_m24", uncompressed)
simplePlot("m14_vs_m13", compressed)
simplePlot("ratio_m14_vs_m24", uncompressed)
simplePlot("ratio_m14_vs_m13", compressed)

simplePlot("pt_1", representative)
'''