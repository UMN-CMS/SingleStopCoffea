# Yet another module. I want to take the highest two btag scores from two leading jets and
#     get an average of those. The value should be stored in an output column.
#     the data will be per-event. The new column should be immediately usable in plots.
@define
class BtagDisc(AnalyzerModule):
    """

    """
    input_1: "Ideally 1st rank pt jet, but a jet"
    input_2: "Ideally 2nd rank pt jet, but another jet"
    output_col: Column

    def run(self, columns, params):
        jet1 = columns[self.input_1]
        jet2 = columns[self.input_2]
        columns[self.output_col] = (jet1.btag_score + jet2.btag_score)/2 
        return columns, []

    def inputs(self, metadata):
        return [self.input_col]

    def outputs(self, metadata):
        return [self.output_col]
