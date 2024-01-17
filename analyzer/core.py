from dataclasses import dataclass

from functools import wraps
from collections import namedtuple, defaultdict
from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    List,
    Set,
    Tuple,
    Union,
    Optional,
    Iterable,
)
from graphlib import TopologicalSorter, CycleError

from collections.abc import Collection, Coroutine, Iterator, Sequence

from distributed import Client

from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.analysis_tools import PackedSelection, Weights
from coffea.dataset_tools.apply_processor import DaskOutputType
from coffea.dataset_tools.preprocess import DatasetSpec


import dask_awkward as dak
import awkward as ak

import analyzer.utils as utils
from analyzer.datasets import SampleSet, SampleCollection
from analyzer.histogram_builder import HistogramBuilder

import coffea.dataset_tools as dst

import itertools as it
import dask
import hist.dask as dah
import hist


from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table


class AnalyzerGraphError(Exception):
    def __init__(self, message):
        super().__init__(message)


def iterableNotStr(t):
    return isinstance(t, Iterable) and not isinstance(t, str)


def toSet(x):
    if iterableNotStr(x):
        return set(x)
    else:
        return (x,)


class AnalyzerModule:
    def __init__(self, name, function, depends_on=None, category=None, after=None):
        self.name = name
        self.function = function
        self.depends_on = toSet(depends_on) if depends_on else set()
        self.categories = toSet(category) if category else set()

    def __call__(self, events, analyzer):
        return self.function(events, analyzer)

    def __str__(self):
        return f"AnalyzerModule({self.name}, depends_on={self.depends_on}, catetories={self.categories})"

    def __repr__(self):
        return str(self)


modules = {}
category_after = {
    "main": ["selection", "weights", "category"],
    "weights": ["selection"],
}


def generateTopology(module_list):
    mods = [x.name for x in module_list]
    cats = defaultdict(list)
    for x in module_list:
        for c in x.categories:
            cats[c].append(x.name)
    graph = {}
    for i, module in enumerate(module_list):
        graph[module.name] = module.depends_on
        if i > 0:
            graph[module.name].update(
                {
                    mods[i - 1],
                }
            )

        for c in module.categories:
            for a in category_after.get(c, []):
                graph[module.name].update(set(cats[a]))

        for m in graph[module.name]:
            if m not in mods:
                raise AnalyzerGraphError(
                    f"Module {module.name} depends on {m}, but was this dependency was not supplied"
                )
    return graph


def namesToModules(module_list):
    return [modules[x] for x in module_list]


def sortModules(module_list):
    graph = generateTopology(module_list)
    try:
        ts = TopologicalSorter(graph)
        ret = tuple(ts.static_order())
    except CycleError as e:
        raise AnalyzerGraphError(
            f"Cyclic dependency detected in module specification:\n {' -> '.join(e.args[1])}\n"
            f"You may need to reorder the modules."
        )
    return namesToModules(ret)


def analyzerModule(name, depends_on=None, categories=None):
    def decorator(func):
        if name in modules:
            raise KeyError(f"A module already exists with the name {name}")

        modules[name] = AnalyzerModule(name, func, depends_on, categories)
        return func

    return decorator


class AnalyzerError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


@dataclass
class DatasetInput:
    dataset_name: str
    fill_name: str
    coffea_dataset: DatasetSpec

    @staticmethod
    def fromSampleOrCollection(
        sample_or_collection: Union[SampleSet, SampleCollection]
    ):
        return [
            DatasetInput(s.name, s.fillname, s.coffea_dataset)
            for s in sample_or_collection.getAnalyzerSamples()
        ]


@dataclass
class DatasetPreprocessed:
    dataset_input: DatasetInput
    coffea_dataset_split: DatasetSpec

    @staticmethod
    def fromDatasetInput(dataset_input, client, **kwargs):
        out, x = dst.preprocess(dataset_input.coffea_dataset, **kwargs)
        return DatasetPreprocessed(dataset_input, out)

    def getCoffeaDataset(self) -> DatasetSpec:
        return self.coffea_dataset_split[self.dataset_input.dataset_name]


@dataclass
class DatasetDaskRunResult:
    dataset_preprocessed: DatasetPreprocessed
    histograms: Dict[str, dah.Hist]
    raw_events_processed: Any
    run_report: dak.Array


@dataclass
class DatasetRunResult:
    dataset_preprocessed: DatasetPreprocessed
    histograms: Dict[str, hist.Hist]
    raw_events_processed: int
    dataset_run_report: ak.Array

    def getScaledHistograms(self, sample_manager, target_lumi):
        sample = sample_manager[self.dataset_preprocessed.dataset_input.dataset_name]
        weight = sample.getWeight(target_lumi)
        reweighted = sample.n_events / self.raw_events_processed
        final_weight = reweighted * weight
        return {name: final_weight * h for name, h in self.histograms.items()}


def mergeAndWeightResults(results, sample_manager, target_lumi=None):
    return utils.accumulate(
        [x.getScaledHistograms(sample_manager, target_lumi) for x in results]
    )


class DatasetProcessor:
    def __init__(
        self,
        dask_result: DatasetDaskRunResult,
        fill_name: str,
    ):
        self.fill_name = fill_name

        self.dask_result = dask_result
        self.__selection = PackedSelection()
        self.__weights = Weights(None)

        self.histogram_builder = HistogramBuilder(self.weights)

    @property
    def selection(self):
        return self.__selection

    @property
    def histograms(self):
        return self.dask_result.histograms

    @property
    def weights(self):
        return self.__weights

    def applySelection(self, events):
        events = events[self.selection.all(*self.selection.names)]
        return events

    def maybeCreateAndFill(
        self,
        key,
        axis,
        data,
        mask=None,
        name=None,
        description=None,
        auto_expand=True,
    ):
        if key not in self.histograms:
            self.histograms[key] = self.histogram_builder.createHistogram(
                axis, name, description
            )
        self.histogram_builder.fillHistogram(self.histograms[key], data, mask)

    def H(self, *args, **kwargs):
        return self.maybeCreateAndFill(*args, **kwargs)


class Analyzer:
    """
    Represents an analysis, a collection of modules.
    """

    def __init__(self, modules: Iterable[AnalyzerModule], cache: Any):
        self.modules: List[AnalyzerModule] = self.__createAndSortModules(*modules)
        self.cache = cache

        self.__dataset_ps: Dict[str, DatasetProcessingState] = {}
        self.__run_reports: Dict[str, dak.Array] = {}

    def __createAndSortModules(self, *module_names):
        m = namesToModules(module_names)
        t = generateTopology(m)
        modules = sortModules(m)
        return modules

    def getDatasetFutures(self, dsprep: DatasetPreprocessed) -> DatasetDaskRunResult:
        dataset_name = dsprep.dataset_input.dataset_name
        files = dsprep.getCoffeaDataset()["files"]
        events, report = NanoEventsFactory.from_root(
            files,
            schemaclass=NanoAODSchema,
            uproot_options={"allow_read_errors_with_report": True},
            persistent_cache=self.cache,
        ).events()

        daskres = DatasetDaskRunResult(dsprep, {}, dak.num(events, axis=0), report)
        dataset_analyzer = DatasetProcessor(daskres, dsprep.dataset_input.fill_name)
        num = (dak.num(events, axis=0),)
        for m in self.modules:
            print(f"Adding module {m}")
            test = m(events, dataset_analyzer)
            events, dataset_analyzer = test
        return daskres

    def execute(self, futures: Iterable[DatasetDaskRunResult], client: Client):
        futures = list(futures)
        dsk = [
            [
                x.histograms,
                x.raw_events_processed,
                x.run_report,
            ]
            for x in futures
        ]
        res = client.compute(dsk)
        gathered = client.gather(res)
        return {
            x.dataset_preprocessed.dataset_input.dataset_name: DatasetRunResult(
                x.dataset_preprocessed, h, r, rep
            )
            for x, (h, r, rep) in zip(futures, gathered)
        }
