import hist
import awkward as ak

dataset_axis = hist.axis.StrCategory(
    [], growth=True, name="dataset", label="Primary dataset"
)

mass_axis = hist.axis.Regular(150, 0, 3000, name="mass", label=r"$m$ [GeV]")
pt_axis = hist.axis.Regular(150, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")
ht_axis = hist.axis.Regular(300, 0, 3000, name="ht", label=r"$HT$ [GeV]")
dr_axis = hist.axis.Regular(20, 0, 5, name="dr", label=r"$\Delta R$")
eta_axis = hist.axis.Regular(20, -5, 5, name="eta", label=r"$\eta$")
phi_axis = hist.axis.Regular(50, 0, 4, name="phi", label=r"$\phi$")
nj_axis = hist.axis.Regular(10, 0, 10, name="nj", label=r"$n_{j}$")
tencountaxis = hist.axis.Regular(10, 0, 10, name="Number", label=r"Number")
b_axis = hist.axis.Regular(5, 0, 5, name="nb", label=r"$n_{b}$")

def makeHistogram(
    axis, dataset, data, weights, name=None, description=None, drop_none=True
):
    if isinstance(axis, list):
        h = hist.Hist(dataset_axis, *axis, storage="weight", name=name)
    else:
        h = hist.Hist(dataset_axis, axis, storage="weight", name=name)
    setattr(h, "description", description)
    if isinstance(axis, list):
        ret = h.fill(dataset, *data, weight=weights)
    else:
        ret = h.fill(dataset, ak.to_numpy(data), weight=weights)
    return ret

