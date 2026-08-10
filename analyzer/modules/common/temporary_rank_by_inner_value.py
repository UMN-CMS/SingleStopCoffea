from analyzer.core.analysis_modules import AnalyzerModule
import re

from analyzer.core.columns import addSelection
from analyzer.core.columns import Column
from analyzer.utils.structure_tools import flatten
from analyzer.core.analysis_modules import ParameterSpec, ModuleParameterSpec
import awkward as ak
import itertools as it
from attrs import define, field, evolve
from .axis import RegularAxis
from .histogram_builder import makeHistogram
from analyzer.core.adl import ADLBlock, ADLStatement


import correctionlib
import logging


@define
class PromoteInnerValIndex(AnalyzerModule):
    """

    Parameters
    ----------
    input_col : Column
        Column containing a nested collection (e.g. ``N × M`` objects).
    input_inner_rank: int
        An inner descriptor which would have an integer value to index by.
    output_col : Column
        Column where the selected elements will be stored.
    index : int, optional
        Index of the element to promote from each nested collection,
        by default ``0`` (leading element).

    """

    input_col: Column
    input_inner_rank: int
    output_col: Column
    index: int = 0

    def run(self, columns, params):
        columns[self.output_col] = ak.pad_none(columns[self.input_col], self.index + 1, axis=1)[:, self.index]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

