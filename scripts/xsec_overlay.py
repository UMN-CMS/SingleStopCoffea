import numpy as np
import mplhep
import matplotlib as mpl
from analyzer.postprocessing.plots.mplstyles import loadStyles
from matplotlib import pyplot as plt
import yaml

xsec_stop = {
    #200.0: 170000.0,
    #300.0: 40000.0,
    #500.0: 8000.0,
    #700.0: 2200.0,
    #800.0: 1300.0,
    #900.0: 770.0,
    #1000.0: 480.0,
    #1100.0: 310.0,
    #1200.0: 210.0,
    #1300.0: 150.0,
    #1400.0: 100.0,
    #1500.0: 73.0,
    1800.0: 28.0,
    2000.0: 16.0,
    2500.0: 4.2,
    3000.0: 1.3,
}
loadStyles()
mpl.use("Agg")
# Load the xsec_z_obs and xsec_z_exp data from a YAML file
with open("scripts/xsec_overlay.yaml", "r") as file:
    data = yaml.safe_load(file)

mass_points = [i["value"] for i in data["independent_variables"][0]['values']]

xsec_z_obs = [i["value"] for i in data["dependent_variables"][0]['values']]
xsec_z_exp = [i["value"] for i in data["dependent_variables"][1]['values']]
xsec_z_exp_1_std_dev = [(i["errors"][0]["asymerror"]['plus'], i["errors"][0]["asymerror"]["minus"]) for i in data["dependent_variables"][1]['values']]
xsec_z_exp_2_std_dev = [(i["errors"][1]["asymerror"]['plus'], i["errors"][1]["asymerror"]["minus"]) for i in data["dependent_variables"][1]['values']]

#new fill_between dict for xsec_z_exp_1_std_dev
xsec_z_exp_fill_between_1std = [] 
xsec_z_exp_fill_between_2std = [] 
for index,value in enumerate(xsec_z_exp):
    xsec_z_exp_fill_between_1std.append((value + xsec_z_exp_1_std_dev[index][0], value + xsec_z_exp_1_std_dev[index][1]))
    xsec_z_exp_fill_between_2std.append((value + xsec_z_exp_2_std_dev[index][0], value + xsec_z_exp_2_std_dev[index][1]))

plt.fontweight = "bold"
mplhep.style.use("CMS")
mplhep.cms.label(data=True, lumi=137) 
plt.plot(mass_points, xsec_z_obs, label="Z' Obs", color='black', marker='.', markersize=10,linewidth=3)
plt.plot(mass_points, xsec_z_exp, label="Z' (Expected)", color='black',linestyle='dotted',linewidth=3)
plt.fill_between(mass_points, [y[0] for y in xsec_z_exp_fill_between_1std], [y[1] for y in xsec_z_exp_fill_between_1std], color='#00cc00', alpha=1, label="±1 std. deviation",zorder=1)
plt.fill_between(mass_points, [y[0] for y in xsec_z_exp_fill_between_2std], [y[1] for y in xsec_z_exp_fill_between_2std], color='#ffcc00', alpha=1, label="±2 std. deviation",zorder=0)
plt.plot(list(xsec_stop.keys()), list(xsec_stop.values()), label="Stop", color='red', linewidth=3)
plt.yscale("log")
plt.xlabel("Resonance Mass (GeV)")
plt.ylabel("$\sigma$ * Br(b b-bar) (fb)")
plt.ylim(0.1, 4500)
plt.xlim(1800, 8000)
plt.grid()
plt.legend(title="95% CL Upper Limits", alignment='left', title_fontsize=22)
plt.tight_layout()
plt.savefig("scripts/xsec_overlay.pdf")