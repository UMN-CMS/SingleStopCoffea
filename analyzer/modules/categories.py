import awkward as ak
from analyzer.core import analyzerModule, ModuleType
from analyzer.modules.axes import *
import hist



@analyzerModule("dataset_category", ModuleType.Categories)
def datasetCategory(events, data):
    return (dataset_axis, data["CatDataset"])

@analyzerModule("njets_category", ModuleType.Categories)
def njetCategory(events,data):
    a = hist.axis.IntCategory([4, 5, 6], name="number_jets", label="NJets")
    return (a, ak.num(events.good_jets, axis=1))
