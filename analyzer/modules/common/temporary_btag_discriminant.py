# Yet another module. I want to take the highest two btag scores from two leading jets and
#     get an average of those. The value should be stored in an output column.
#     the data will be per-event. The new column should be immediately usable in plots.
import itertools as it
import awkward as ak
import correctionlib
import correctionlib.convert
from analyzer.core.columns import Column
from attrs import define, field
import correctionlib
from analyzer.core.datasets import SampleType
from analyzer.core.analysis_modules import (
    AnalyzerModule,
    MetadataExpr,
    ParameterSpec,
    ModuleParameterSpec,
    IsSampleType,
)

@define
class BtagDisc(AnalyzerModule):
    """
        Note: For this module to work, the promote index module must be called to promote
                BY the btag score.

        input_col_first_rank: Column 
                from promote index, highest bscore
        input_col_second_rank: Column 
                from another promote index, second highest bscore
        output_col: Column
                The desired name for the column containing (1stB+2ndB)/2 values       
    """
    input_col_first_rank: Column 
    input_col_second_rank: Column 
    output_col: Column
                
    def run(self, columns, params):                                                               
        btag1 = self.input_col_first_rank
        btag2 = self.input_col_second_rank

        btag1_values = btag1.extract(columns)
        btag2_values = btag2.extract(columns)
        columns[self.output_col] = (btag1_values + btag2_values)/2
        return columns, {}

    def inputs(self, metadata):
        return [self.input_col_first_rank, self.input_col_second_rank]

    def outputs(self, metadata):
        return [self.output_col]
