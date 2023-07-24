import hist

dataset_axis = hist.axis.StrCategory(
    [], growth=True, name="dataset", label="Primary dataset"
)

mass_axis = hist.axis.Regular(100, 0, 3000, name="mass", label=r"$m$ [GeV]")
pt_axis = hist.axis.Regular(100, 0, 1500, name="pt", label=r"$p_{T}$ [GeV]")
ht_axis = hist.axis.Regular(100, 0, 3000, name="ht", label=r"$HT$ [GeV]")
dr_axis = hist.axis.Regular(20, 0, 5, name="dr", label=r"$\Delta R$")
eta_axis = hist.axis.Regular(20, -5, 5, name="eta", label=r"$\eta$")
phi_axis = hist.axis.Regular(50, 0, 4, name="phi", label=r"$\phi$")
nj_axis = hist.axis.Regular(10, 0, 10, name="nj", label=r"$n_{j}$")
tencountaxis = hist.axis.Regular(10, 0, 10, name="Number", label=r"Number")
b_axis = hist.axis.Regular(5, 0, 5, name="nb", label=r"$n_{b}$")
bool_axis = hist.axis.IntCategory([0,1], name="truefalse", label=r"$n_{b}$")
