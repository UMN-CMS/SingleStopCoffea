import awkward as ak
from analyzer.core import analyzerModule
from analyzer.modules.axes import *
import hist


@analyzerModule("dataset_category", categories="axis_cat", depends_on=["objects"])
def datasetCategory(events, analyzer):
    analyzer.histogram_builder.addCategory(dataset_axis, analyzer.fill_name)
    return events


@analyzerModule("njets_category", depends_on=["objects"], categories="axis_cat")
def njetCategory(events, analyzer):
    #a = hist.axis.IntCategory([4, 5, 6], name="number_jets", label="NJets")
    return events
