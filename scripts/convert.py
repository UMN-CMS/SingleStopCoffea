import pickle as pkl
import sys
sys.path.append(".")
import hist
import uproot

from pathlib import Path


file_name = "all_hists.pkl"
data = pkl.load(open(file_name, "rb"))
histos = data["histograms"]

root_output= uproot.recreate("output.root")

def convertToRoot(h, name, dset_axis = "dataset"):
    for x in h.axes[dset_axis]:
        newhist = h[{dset_axis: x}]
        root_output[f"{x}_{name}"] = newhist.to_numpy()




for h in histos:
    convertToRoot(histos[h], h)

