from coffea.nanoevents import NanoAODSchema, NanoEventsFactory
import uproot
import pprint
#from .preprocessing_tools import preprocess

fname = "root://cmseos.fnal.gov//store/user/dmahon/condor/RPVSingleStopMC/NANOAOD-ALL/NANOAOD-1000_100.root"

events = NanoEventsFactory.from_root(
    {fname: "Events"},
    delayed=False,
).events()

tau2_coffea_bad = events["FatJet"].tau2
tau2_coffea_good = events["FatJet"]["tau2"]
coffea_fatjet = events["FatJet"][0]

uproot_file = uproot.open(fname)
tau2_uproot = uproot_file["Events"]["FatJet_tau2"].array()
fat_jet_keys = [i for i in uproot_file["Events"].keys() if "FatJet_" in i]
uproot_fatjet = uproot_file["Events"].arrays(filter_name="FatJet_*")[0].to_list()

for key in uproot_fatjet.keys():
    uproot_fatjet[key] = uproot_fatjet[key][0]

print("Result using Coffea")
print('Good, using ["tau2"]:', tau2_coffea_good, '\nBad, using .tau2:', tau2_coffea_bad)
print('vs')
print("Result using Uproot")
print(tau2_uproot)
raise Exception()
print("Coffea FatJet:")
coffea_fatjet[0].show(limit_rows=70)
print('vs')
print("Uproot FatJet:")
pprint.pp(uproot_fatjet)