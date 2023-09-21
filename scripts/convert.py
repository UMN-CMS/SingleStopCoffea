import pickle as pkl
import sys

sys.path.append(".")
import hist
import uproot

from pathlib import Path
import argparse
from rich.progress import track





def convertToRoot(h, name, out, dset_axis="dataset"):
    for x in h.axes[dset_axis]:
        newhist = h[{dset_axis: x}]
        out[f"{x}_{name}"] = newhist.to_numpy()




def main():
    parser = argparse.ArgumentParser(
        description="Convert an analyzer output file to root"
    )
    parser.add_argument("input", type=str)
    parser.add_argument("output", type=str)
    args= parser.parse_args()
    data = pkl.load(open(args.input, "rb"))
    root_output = uproot.recreate(args.output)
    histos = data["histograms"]
    for h in track(histos, description = "Converting..."):
        convertToRoot(histos[h], h, root_output)


if __name__ == "__main__":
    main()
