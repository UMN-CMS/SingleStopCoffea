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
class SortByBtagScore(AnalyzerModule):
    input_col: Column
    output_col: Column
    btag_field: str

    def run(self, columns, params):
        jets = self.input_col.extract(columns)
        order = ak.argsort(jets[self.btag_field], axis=1, ascending=False)
        columns[self.output_col] = jets[order]
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]

@define
class SelectTopN(AnalyzerModule):
      input_col: Column
      output_col: Column
      sort_field: str
      n: int

      def run(self, columns, params):
          jets = self.input_col.extract(columns)
          order = ak.argsort(jets[self.sort_field], axis=1, ascending=False)
          sorted_jets = jets[order]
          columns[self.output_col] = sorted_jets[:, :self.n]
          return columns, []

      def inputs(self, metadata):
          return [self.input_col]

      def outputs(self, metadata):
          return [self.output_col]
