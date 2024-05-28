import sys
from pathlib import Path

sys.path.append(str(Path(".").resolve()))

import argparse

import matplotlib as mpl
import matplotlib.pyplot as plt
import torch
from analyzer.plotting.mplstyles import loadStyles
from fitting.plot_tools import createSlices, getPolyFromSquares, makeSquares, simpleGrid
from fitting.regression import getNormalizationTransform

mpl.use("Agg")


def loadModelFromDirectory(path):
    path = Path(path)
    d = torch.load(path)
    m = d["model"]
    train_data = d["train_data"]
    test_data = d["test_data"]
    nt = getNormalizationTransform(train_data)
    m.eval()
    return m, nt, train_data, test_data


def parseArgs():
    parser = argparse.ArgumentParser(description="")
    parser.add_argument("-o", "--output", type=str, required=True)
    parser.add_argument("-t", "--target", type=float, nargs=2, required=True)
    parser.add_argument("infile", type=str)
    return parser.parse_args()


def getCovariances(target, points, covariance_function, transform):
    t = transform.transform_x.transformData(torch.tensor([target]))
    p = transform.transform_x.transformData(points)
    cv = covariance_function(t, p)
    x = torch.squeeze(cv.to_dense().detach())
    return x


def plotCovariance(ax, target, covariances, edges, points):
    simpleGrid(ax, edges, points, covariances)
    ax.plot([target[0]], [target[1]], 'o', color="red", markersize=10)
    ax.set_xlabel("$m_4$ [GeV]", fontsize=24)
    ax.set_ylabel(r"$m_{3} / m_{4}$", fontsize=24)
    if hasattr(ax, "cax"):
        cax = ax.cax
        cax.set_ylabel("$k(x_1,x_2)$")



def main():
    args = parseArgs()
    f = Path(args.infile)
    o = Path(args.output)
    m, nt, tr, tt = loadModelFromDirectory(f)
    target = args.target
    cv = getCovariances(target, tt.X, m.covar_module, nt)
    fig, ax = plt.subplots()
    plotCovariance(ax, target, cv, tt.E, tt.X)

    o.parent.mkdir(exist_ok=True, parents=True)
    fig.savefig(o)



if __name__ == "__main__":
    loadStyles()
    main()
