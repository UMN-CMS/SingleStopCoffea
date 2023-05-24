import pickle 
import uproot
import hist
import math
import itertools as it
import re

def getHistograms(path):
    with open(path, "rb") as f:
        r = pickle.load(f)
    return r


all_hists = getHistograms("output.pkl")




def makeOptimized(h_sig, h_bkg, opt_axis = None, disc_axis=None, start=0, stop=0, step=0):
    cur_max = None
    opt_axis = next(x for x in h_sig.axes if x.name==opt_axis)
    disc_axis = next(x for x in h_sig.axes if x.name==disc_axis)
    cur_best = 0
    for i in range(start,stop,step):
        bkg_temp = h_bkg[{opt_axis.name : slice(hist.loc(i), len(opt_axis))}]
        sig_temp = h_sig[{opt_axis.name : slice(hist.loc(i), len(opt_axis))}]
        bkg_count = bkg_temp.sum()
        sig_count = sig_temp.sum()
        if bkg_count.value > 0 :
            new = sig_count.value / math.sqrt(bkg_count.value)
            if new > cur_best:
                cur_best = new
                best_hists = (sig_temp.project(disc_axis.name), bkg_temp.project(disc_axis.name))
    return best_hists


f = uproot.recreate("example.root")
#for n,h in all_hists.items():
#    for d in h.axes[0]:
#        f[f"{d}_{n}"] = h[d,...]
#
#twodmasses= [x for x in all_hists if "vs" in x and "m" in x]
#for h in twodmasses:
#    hi = all_hists[h]
#    hb = hi["QCD2018", ...]

signals = [x for x in all_hists["HT"].axes[0] if "signal" in x ]
for sig in signals:
    m = re.match(r"signal_(\d+)_(\d+)_Skim", sig)
    m1,m2=m.group(1,2)
    m1,m2 = int(m1), int(m2)
    if abs(m1 - m2) <= 200:
        target = "m03_vs_m04"
        a1="03"
    else:
        target = "m14_vs_m04"
        a1="14"
    hi = all_hists[target]
    hs = hi[sig, ...]
    hb = hi["QCD2018", ...]
    bs,bb = makeOptimized(hs, hb, f"mass_{a1}", "mass_04", 500,2000, 20)
    f[f"{sig}_{target}_opt_{a1}_proj_04"] = bs
    f[f"QCD2018_{sig}_{target}_opt_{a1}_proj_04"] = bb
