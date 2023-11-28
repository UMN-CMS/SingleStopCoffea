import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
from analyzer.plotting.core_plots import *
import warnings
import itertools as it
import json

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = ["Skim_QCDInclusive2018"]
compressed = [
    f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400", "2000_1400")
]
uncompressed = [f"signal_312_{p}" for p in ("2000_1400", "1200_400", "1500_900")]
both = compressed + uncompressed
representative = [f"signal_312_{p}" for p in ("2000_1900", "1200_400", "1500_900")]


plotter = Plotter("everymass.pkl", "figures", default_backgrounds=backgrounds)
histos = plotter.histos

ret = {}
for x, y, z in it.combinations(list(range(6)), 3):
    name = f"charg_mass_{x}_{y}_{z}"
    h = histos[name]
    ret[name] = {}
    for s in h.axes[0]:
        sh = h[s, ...]
        centers = sh.axes[0].centers
        weights = sh.values()
        mean = np.average(centers, weights=weights)
        std = np.sqrt(np.average((mean - centers) ** 2, weights=weights))
        ret[name][s] = {"mean": mean, "std": std}
    continue
    plotter(
        f"charg_mass_{x}_{y}_{z}",
        compressed,
        [],
        normalize=True,
        scale="linear",
        sig_style="hist",
        add_name="compressed",
    )
    plotter(
        f"charg_mass_{x}_{y}_{z}",
        uncompressed,
        [],
        normalize=True,
        scale="linear",
        sig_style="hist",
        add_name="uncompressed",
    )

json.dump(ret, open("allmasscalcs.json", "w"), indent=2)


def renderAsLatex(table, hlines=None):
    hlines = hlines or []
    ret = ""
    rowstrs = []
    for i, row in enumerate(table):
        colstrs = [
            str(c) if not isinstance(c, tuple) else f"\cellcolor{{{c[0]}!25}}{c[1]}"
            for c in row
        ]
        s = " & ".join(colstrs)
        if i in hlines:
            s = "\\hline " + s
        rowstrs.append(s)

    ret = " \\\\\n".join(rowstrs)
    ret = f"\\begin{{tabular}}{{{'|' + '|'.join(['c']*len(table[0])) + '|'}}}\n" + ret
    ret += f"\n\\end{{tabular}}"
    return ret


ltable = [
    ["Combination"] + [",".join(m.split("_")[2:4]) for m in compressed + uncompressed]
] + [
    [
        ",".join(x.split("_")[2:5]),
        *[
            f"{int(v[y]['mean'])}({int(v[y]['std'])})"
            for y in compressed + uncompressed
        ],
    ]
    for x, v in ret.items()
]
print(renderAsLatex(ltable, hlines=[1]))
