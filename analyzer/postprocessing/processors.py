import abc
from enum import Enum, auto
import functools as ft
import logging
from pathlib import Path
from typing import Literal, ClassVar
from .tex import renderTemplate

import yaml

import pydantic as pyd
from analyzer.configuration import CONFIG
from analyzer.core.specifiers import SectorSpec, SectorParams

from .grouping import (
    SectorGroupSpec,
    createSectorGroups,
    doFormatting,
    SectorGroupParameters,
    groupsMatch,
    groupBy,
)
from .plots.export_hist import exportHist
from .plots.plots_1d import PlotConfiguration, plotOne, plotRatio, plotStrCat
from .plots.plots_2d import plot2D, plotRatio2D, plotRatio3D
from .registry import registerPostprocessor
from .split_histogram import Mode
from .style import Style, StyleSet

logger = logging.getLogger(__name__)
StyleLike = Style | str


class PostprocessCatalogueEntry(pyd.BaseModel):
    processor_name: str
    identifier: str
    path: str
    sector_group: SectorGroupParameters
    sector_params: list[SectorParams]


postprocess_catalog = pyd.TypeAdapter(list[PostprocessCatalogueEntry])


class PostProcessorType(Enum):
    Normal = auto()
    Accumulator = auto()


class BasePostprocessor(abc.ABC):
    postprocessor_type: ClassVar[PostProcessorType] = PostProcessorType.Normal
    name: str

    @abc.abstractmethod
    def getExe(self, results):
        pass

    def getNeededHistograms(self):
        return []

    def init(self):
        if isinstance(self.style_set, str):
            print("Loading style set")
            config_path = Path(CONFIG.STYLE_PATH) / self.style_set
            with open(config_path, "r") as f:
                d = yaml.safe_load(f)
            self.style_set = StyleSet(**d)


@registerPostprocessor
class Histogram1D(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    to_process: SectorSpec
    style_set: str | StyleSet
    output_name: str
    grouping: SectorGroupSpec

    scale: Literal["log", "linear"] = "linear"
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getNeededHistograms(self):
        return self.histogram_names

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, self.grouping)
        ret = []
        items = []
        for histogram in self.histogram_names:
            for sector_group in r:
                pc = self.plot_configuration.makeFormatted(sector_group.all_parameters)
                hists = sector_group.histograms(histogram)
                if not hists:
                    continue
                output = doFormatting(
                    self.output_name,
                    sector_group.all_parameters,
                    histogram_name=histogram,
                )

                ret.append(
                    ft.partial(
                        plotOne,
                        hists,
                        sector_group.all_parameters,
                        output,
                        scale=self.scale,
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=pc,
                    )
                )

                items.append(
                    PostprocessCatalogueEntry(
                        processor_name=self.name,
                        identifier=histogram,
                        path=output,
                        sector_group=sector_group,
                        sector_params=[x.sector_params for x in sector_group.sectors],
                    )
                )
        return ret, items


@registerPostprocessor
class ExportHists(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    to_process: SectorSpec
    grouping: SectorGroupSpec
    output_name: str

    def getNeededHistograms(self):
        return self.histogram_names

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, self.grouping)
        ret = []
        items = []
        for histogram in self.histogram_names:
            for sector_group in r:
                hists = sector_group.histograms(histogram)
                output = doFormatting(
                    self.output_name,
                    sector_group.all_parameters,
                    histogram_name=histogram,
                )
                if not hists:
                    continue
                ret.append(
                    ft.partial(
                        exportHist,
                        hists[0],
                        sector_group.all_parameters,
                        output,
                    )
                )
        return ret, items

    def init(self):
        return


@registerPostprocessor
class Histogram2D(BasePostprocessor, pyd.BaseModel):

    histogram_names: list[str]
    to_process: SectorSpec
    style_set: str | StyleSet
    grouping: SectorGroupSpec
    output_name: str = "{histogram_name}"

    axis_options: dict[str, Mode | str | int] | None = None
    normalize: bool = False

    color_scale: Literal["log", "linear"] = "linear"

    plot_configuration: PlotConfiguration | None = None

    def getNeededHistograms(self):
        return self.histogram_names

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, self.grouping)
        ret = []
        items = []
        for histogram in self.histogram_names:
            for sector_group in r:
                output = doFormatting(
                    self.output_name,
                    sector_group.all_parameters,
                    histogram_name=histogram,
                )
                ret.append(
                    ft.partial(
                        plot2D,
                        sector_group.histograms(histogram)[0],
                        sector_group.all_parameters,
                        output,
                        style_set=self.style_set,
                        normalize=self.normalize,
                        plot_configuration=self.plot_configuration,
                        color_scale=self.color_scale,
                    )
                )
                items.append(
                    PostprocessCatalogueEntry(
                        processor_name=self.name,
                        identifier=histogram,
                        path=output,
                        sector_group=sector_group,
                        sector_params=[x.sector_params for x in sector_group.sectors],
                    )
                )
        return ret, items


@registerPostprocessor
class PlotCutflow(BasePostprocessor, pyd.BaseModel):
    to_process: SectorSpec
    style_set: str | StyleSet
    grouping: SectorGroupSpec
    output_name: str
    plot_types: list[str] = ["cutflow", "one_cut", "n_minus_one"]

    normalize: bool = False
    table_mode: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getExe(self, results):
        sectors = [x for x in results if self.to_process.passes(x.sector_params)]
        r = createSectorGroups(sectors, self.grouping)
        ret, items = [], []

        for sector_group in r:
            for pt in self.plot_types:
                pc = self.plot_configuration.makeFormatted(sector_group.all_parameters)
                output = doFormatting(
                    self.output_name,
                    sector_group.all_parameters,
                    histogram_name=pt,
                )
                ret.append(
                    ft.partial(
                        plotStrCat,
                        pt,
                        sector_group.parameters,
                        sector_group.sectors,
                        output,
                        self.style_set,
                        table_mode=self.table_mode,
                        normalize=self.normalize,
                        plot_configuration=pc,
                    )
                )
                items.append(
                    PostprocessCatalogueEntry(
                        processor_name=self.name,
                        identifier=str(sector_group),
                        path=output,
                        sector_group=sector_group,
                        sector_params=[x.sector_params for x in sector_group.sectors],
                    )
                )
        return ret, items


@registerPostprocessor
class RatioPlot(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    numerator: SectorGroupSpec
    denominator: SectorGroupSpec
    to_process: SectorSpec
    style_set: str | StyleSet
    output_name: str
    match_fields: list[str]
    scale: Literal["log", "linear"] = "linear"
    axis_options: dict[str, Mode | str | int] | None = None
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None
    ratio_ylim: tuple[float, float] = (0, 2)
    ratio_hlines: list[float] = pyd.Field(default_factory=lambda: [1.0])
    ratio_height: float = 1.5
    ratio_type: Literal["poisson", "poisson-ratio", "efficiency"] = "poisson"

    def getNeededHistograms(self):
        return self.histogram_names

    def getExe(self, results):
        results = [x for x in results if self.to_process.passes(x.sector_params)]
        gnums = createSectorGroups(results, self.numerator)
        gdens = createSectorGroups(results, self.denominator)
        ret, items = [], []
        for histogram in self.histogram_names:
            for den_group in gdens:
                try:
                    num_group = list(
                        x for x in gnums if groupsMatch(den_group, x, self.match_fields)
                    )
                    if len(num_group) != 1:
                        raise KeyError(f"Too many groups")
                    num_group = next(iter(num_group))
                except StopIteration:
                    raise KeyError(f"Could not find group")

                # print(
                #     f"Denominator group\n{den_group}\n matched with numerator group\n{num_group}"
                # )

                dh = den_group.histograms(histogram)
                if not dh:
                    continue
                if len(dh) != 1:
                    raise RuntimeError
                output = doFormatting(
                    self.output_name, den_group.all_parameters, histogram_name=histogram
                )
                ret.append(
                    ft.partial(
                        plotRatio,
                        dh[0],
                        num_group.histograms(histogram),
                        output,
                        self.style_set,
                        normalize=self.normalize,
                        ratio_ylim=self.ratio_ylim,
                        ratio_type=self.ratio_type,
                        scale=self.scale,
                        ratio_hlines=self.ratio_hlines,
                        ratio_height=self.ratio_height,
                        plot_configuration=self.plot_configuration,
                    )
                )
                items.append(
                    PostprocessCatalogueEntry(
                        processor_name=self.name,
                        identifier=histogram,
                        path=output,
                        sector_group=den_group,
                        sector_params=[
                            x.sector_params
                            for x in [*den_group.sectors, *num_group.sectors]
                        ],
                    )
                )
        return ret, items


def filterCatalog(catalog, fields):
    return groupBy(catalog, fields, data_acquire=lambda x: x.model_dump())


@registerPostprocessor
class DocRender(BasePostprocessor, pyd.BaseModel):
    postprocessor_type: ClassVar[PostProcessorType] = PostProcessorType.Accumulator
    template: str
    catalog_paths: list[str]
    doc_level_group: list[str]
    internal_group: list[str]
    output: str

    def getExe(self, results):
        catalog = []
        for path_name in self.catalog_paths:
            for path in Path(".").glob(path_name):
                with open(path, "r") as f:
                    catalog += postprocess_catalog.validate_json(f.read())
        ret = []
        for k, top_level in filterCatalog(catalog, self.doc_level_group):
            data = {"doc_level_parameters": k, "groups": []}
            for i, group_level in filterCatalog(top_level, self.internal_group):
                data["groups"].append(
                    {"params": i, "items": [x.model_dump() for x in group_level]}
                )
            output = doFormatting(self.output, k)
            ret.append(ft.partial(renderTemplate, self.template, output, data))
        return ret

    def init(self):
        pass


@registerPostprocessor
class RatioPlot2D(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    numerator: SectorGroupSpec
    denominator: SectorGroupSpec
    to_process: SectorSpec
    style_set: str | StyleSet
    output_name: str
    match_fields: list[str]
    scale: Literal["log", "linear"] = "linear"
    axis_options: dict[str, Mode | str | int] | None = None
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None

    def getNeededHistograms(self):
        return self.histogram_names

    def getExe(self, results):
        results = [x for x in results if self.to_process.passes(x.sector_params)]
        gnums = createSectorGroups(results, self.numerator)
        gdens = createSectorGroups(results, self.denominator)
        ret, items = [], []
        for histogram in self.histogram_names:
            for den_group in gdens:
                try:
                    num_group = list(
                        x for x in gnums if groupsMatch(den_group, x, self.match_fields)
                    )
                    if len(num_group) != 1:
                        raise KeyError(f"Too many groups")
                    num_group = next(iter(num_group))
                except StopIteration:
                    raise KeyError(f"Could not find group")

                # print(
                #     f"Denominator group\n{den_group}\n matched with numerator group\n{num_group}"
                # )

                dh = den_group.histograms(histogram)
                if not dh:
                    continue
                if len(dh) != 1:
                    raise RuntimeError
                output = doFormatting(
                    self.output_name, den_group.all_parameters, histogram_name=histogram
                )
                ret.append(
                    ft.partial(
                        plotRatio2D,
                        dh[0],
                        num_group.histograms(histogram),
                        output,
                        self.style_set,
                        normalize=self.normalize,
                        color_scale=self.scale,
                        plot_configuration=self.plot_configuration,
                    )
                )
                items.append(
                    PostprocessCatalogueEntry(
                        processor_name=self.name,
                        identifier=histogram,
                        path=output,
                        sector_group=den_group,
                        sector_params=[
                            x.sector_params
                            for x in [*den_group.sectors, *num_group.sectors]
                        ],
                    )
                )
        return ret, items

@registerPostprocessor
class TriggerHist3D(BasePostprocessor, pyd.BaseModel):
    histogram_names: list[str]
    numerator: SectorGroupSpec
    denominator: SectorGroupSpec
    to_process: SectorSpec
    style_set: str | StyleSet
    output_name: str
    match_fields: list[str]
    scale: Literal["log", "linear"] = "linear"
    axis_options: dict[str, Mode | str | int] | None = None
    normalize: bool = False
    plot_configuration: PlotConfiguration | None = None
    def getNeededHistograms(self):
        return self.histogram_names

    def getExe(self, results):
        results = [x for x in results if self.to_process.passes(x.sector_params)]
        gnums = createSectorGroups(results, self.numerator)
        gdens = createSectorGroups(results, self.denominator)
        ret, items = [], []
        for histogram in self.histogram_names:
            for den_group in gdens:
                try:
                    num_group = list(
                        x for x in gnums if groupsMatch(den_group, x, self.match_fields)
                    )
                    if len(num_group) != 1:
                        raise KeyError(f"Too many groups")
                    num_group = next(iter(num_group))
                except StopIteration:
                    raise KeyError(f"Could not find group")

                # print(
                #     f"Denominator group\n{den_group}\n matched with numerator group\n{num_group}"
                # )

                dh = den_group.histograms(histogram)
                if not dh:
                    continue
                if len(dh) != 1:
                    raise RuntimeError
                output = doFormatting(
                    self.output_name, den_group.all_parameters, histogram_name=histogram
                )
                ret.append(
                    ft.partial(
                        plotRatio3D,
                        dh[0],
                        num_group.histograms(histogram),
                        output,
                        self.style_set,
                        normalize=self.normalize,
                        color_scale=self.scale,
                        plot_configuration=self.plot_configuration,
                    )
                )
                # ret.append(
                #     ft.partial(
                #         plot4d_efficiency,
                #         dh[0],
                #         num_group.histograms(histogram),
                #         output,
                #         self.style_set,
                #         normalize=self.normalize,
                #         color_scale=self.scale,
                #         plot_configuration=self.plot_configuration,
                #     )
                # )
                items.append(
                    PostprocessCatalogueEntry(
                        processor_name=self.name,
                        identifier=histogram,
                        path=output,
                        sector_group=den_group,
                        sector_params=[
                            x.sector_params
                            for x in [*den_group.sectors, *num_group.sectors]
                        ],
                    )
                )
        return ret, items