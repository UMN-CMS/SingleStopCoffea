import concurrent.futures as cf
import functools as ft
import logging
from pathlib import Path
from typing import Optional, Union, Literal
from analyzer.datasets import SampleId
from rich import print

import yaml

import pydantic as pyd
from analyzer.configuration import CONFIG
from analyzer.core import AnalysisResult
from analyzer.core.specifiers import SectorSpec
from rich.progress import Progress, track

from .plots.export_hist import exportHist
from .plots.plots_1d import PlotConfiguration, plotOne, plotRatio, plotStrCat
from .plots.plots_2d import plot2D
from .registry import loadPostprocessors, registerPostprocessor
from .split_histogram import Mode
from .style import Style, StyleSet
from .utils import createSectorGroups

logger = logging.getLogger(__name__)
StyleLike = Union[Style, str]


@registerPostprocessor
class Histogram1D(pyd.BaseModel):

    histogram_names: list[str]
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

    axis_options: Optional[dict[str, Union[Mode, str, int]]] = None
    normalize: bool = False

    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        plotOne,
                        histogram,
                        sector_group.sectors,
                        self.output_name,
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)






def makeTable(sectors,  output_name):
    import csv
    res = []
    for sector in sectors:
        p = sector.sector_params
        print(p)
        cutflow = sector.cutflow_data
        eff = cutflow.raw_passed / sector.raw_processed
        res.append((p.dataset.name, eff))
    print(sorted(res))
        
    # o = doFormatting(output_name, p, histogram_name=histogram)
    # with open(o, 'w') as f:
    #     writer = csv.writer(f)
    #     f.write()
        


@registerPostprocessor
class EfficiencyTable(pyd.BaseModel):
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]
    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{region.region_name}"

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for sector_group in r:
            ret.append(
                ft.partial(
                    makeTable,
                    sector_group.sectors,
                    self.output_name,
                )
            )
        return ret

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)

@registerPostprocessor
class ExportHists(pyd.BaseModel):
    histogram_names: list[str]
    to_export: SectorSpec
    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

    def getExe(self, results):
        sectors = [
            x for x in results.values() if self.to_export.passes(x.sector_params)
        ]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        exportHist,
                        histogram,
                        sector_group.sectors,
                        self.output_name,
                    )
                )
        return ret

    def init(self):
        pass


@registerPostprocessor
class Histogram2D(pyd.BaseModel):

    histogram_names: list[str]
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"

    axis_options: Optional[dict[str, Union[Mode, str, int]]] = None
    normalize: bool = False

    color_scale: Literal["log", "linear"] = "linear"

    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for sector_group in r:
                ret.append(
                    ft.partial(
                        plot2D,
                        histogram,
                        sector_group.sectors[0],
                        self.output_name,
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                        color_scale=self.color_scale
                    )
                )
        return ret

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class PlotCutflow(pyd.BaseModel):
    to_plot: SectorSpec
    style_set: Union[str, StyleSet]

    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    output_name: str = "{histogram_name}"
    normalize: bool = False
    table_mode: bool = False
    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        sectors = [x for x in results.values() if self.to_plot.passes(x.sector_params)]
        r = createSectorGroups(sectors, *self.groupby)
        ret = []
        for sector_group in r:
            ret.append(
                ft.partial(
                    plotStrCat,
                    sector_group.sectors,
                    self.output_name,
                    self.style_set,
                    table_mode=self.table_mode,
                    normalize=self.normalize,
                    plot_configuration=self.plot_configuration,
                )
            )
        return ret

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


@registerPostprocessor
class DatasetRatioPlot(pyd.BaseModel):
    histogram_names: list[str]
    numerator: SectorSpec
    denominator: SectorSpec
    groupby: list[str] = ["dataset.era.name", "region.region_name"]
    style_set: Union[str, StyleSet]
    output_name: str = "{histogram_name}"
    axis_options: Optional[dict[str, Union[Mode, str, int]]] = None
    normalize: bool = False
    plot_configuration: Optional[PlotConfiguration] = None

    def getExe(self, results):
        nums = [x for x in results.values() if self.numerator.passes(x.sector_params)]
        dens = [x for x in results.values() if self.denominator.passes(x.sector_params)]
        gnums = createSectorGroups(nums, *self.groupby)
        gdens = createSectorGroups(dens, *self.groupby)
        ret = []
        for histogram in self.histogram_names:
            for den_group in gdens:
                try:
                    num_group = next(x for x in gnums if den_group.compatible(x))
                except StopIteration:
                    raise KeyError(f"Could not find group {group}")

                ret.append(
                    ft.partial(
                        plotRatio,
                        histogram,
                        num_group.sectors,
                        den_group.sectors[0],
                        self.output_name,
                        self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                    )
                )
        return ret

    def init(self):
        config_path = Path(CONFIG.STYLE_PATH) / "style.yaml"
        with open(config_path, "r") as f:
            d = yaml.safe_load(f)
        self.style_set = StyleSet(**d)


if __name__ == "__main__":
    from analyzer.logging import setup_logging
    from .plots.mplstyles import loadStyles

    loadStyles()

    setup_logging()

    loaded = loadPostprocessors("configurations/eff.yaml")
    # loaded = loadPostprocessors("configurations/post.yaml")
    result = AnalysisResult.fromFile("testout.pkl")
    # result = AnalysisResult.fromFile("testout.pkl")
    # result2 = AnalysisResult.fromFile("results/histograms/2024_10_06.pkl")
    # result2 = AnalysisResult.fromFile("results/histograms/2024_11_05_patched.pkl")
    # result = AnalysisResult.fromFile("test.pkl")
    # print(list(result.results.keys()))
    result = result.getResults(drop_samples=[
        SampleId("QCDInclusive2018", "QCDInclusive2018_HT50to100"),
        SampleId("QCDInclusive2018", "QCDInclusive2018_HT100to200"),
        # SampleId("QCDInclusive2018", "QCDInclusive2018_HT300to500"),
                                             ])
    tasks = []
    for processor in loaded:
        processor.init()
        tasks += processor.getExe(result)

    results = [f() for f in tasks]
    import sys
    sys.exit()
    print("HERE")
    with Progress() as progress:
        task_id = progress.add_task("[cyan]Processing...", total=len(tasks))
        with cf.ProcessPoolExecutor(max_workers=2) as executor:
            results = [executor.submit(f) for f in tasks]
            for i in cf.as_completed(results):
                progress.advance(task_id)
