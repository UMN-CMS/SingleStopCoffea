from dataclasses import dataclass
from enum import Enum
from datetime import datetime
from pathlib import Path
import pickle as pkl
import itertools as it
from graphlib import TopologicalSorter, CycleError
from collections.abc import Collection, Coroutine, Iterator, Sequence
from collections import namedtuple, defaultdict
from functools import wraps
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


from distributed import Client

from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.analysis_tools import PackedSelection, Weights
from coffea.dataset_tools.apply_processor import DaskOutputType
from coffea.dataset_tools.preprocess import DatasetSpec


import hist
import dask
import dask_awkward as dak
import awkward as ak
import hist.dask as dah

import analyzer.utils as utils
from analyzer.datasets import SampleSet, SampleCollection
from analyzer.histogram_builder import HistogramBuilder

import coffea.dataset_tools as dst


from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
from rich.progress import track

from urllib import parse
import logging
import cProfile
import warnings


pr = cProfile.Profile()

logger = logging.getLogger(__name__)


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
    "category": ["selection"],
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
            DatasetInput(s.name, s.setname, s.coffea_dataset)
            for s in sample_or_collection.getAnalyzerSamples()
        ]


@dataclass(eq=True)
class DatasetPreprocessed:
    dataset_input: DatasetInput
    coffea_dataset_split: DatasetSpec

    @staticmethod
    def fromDatasetInput(dataset_input, client, **kwargs):
        out, x = dst.preprocess(dataset_input.coffea_dataset, save_form=False, **kwargs)
        return DatasetPreprocessed(dataset_input, out)

    def getCoffeaDataset(self) -> DatasetSpec:
        return self.coffea_dataset_split


def preprocessBulk(dataset_input: Iterable[DatasetInput], **kwargs):
    mapping = {x.dataset_name: x for x in dataset_input}
    all_inputs = utils.accumulate([x.coffea_dataset for x in dataset_input])
    out, x = dst.preprocess(all_inputs, **kwargs)
    ret = [DatasetPreprocessed(mapping[k], v) for k, v in out.items()]
    return ret


@dataclass
class DatasetDaskRunResult:
    dataset_preprocessed: DatasetPreprocessed
    histograms: Dict[str, dah.Hist]
    raw_events_processed: Any
    run_report: dak.Array

    def getName(self):
        return self.dataset_preprocessed.dataset_input.dataset_name


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

    def update(self, other):
        if self.dataset_preprocessed != other.dataset_preprocessed:
            raise ValueError()
        new_hists = accumulate([self.histograms, other.histograms])
        total_events = self.raw_events_processed + other.raw_events_processed
        report = ak.concat(self.dataset_run_report, other.dataset_run_report)
        result = DatasetRunResult(
            self.dataset_preprocessed, new_hists, total_events, report
        )
        return result

    def getName(self):
        return self.dataset_preprocessed.dataset_input.dataset_name


class DatasetProcessor:
    def __init__(
        self,
        dask_result: DatasetDaskRunResult,
        setname: str,
    ):
        self.setname = setname
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
        name = name or key
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
        logger.info(
            "Will run modules in the following order:\n"
            + "\n".join(f"\t{i+1}. {x.name}" for i, x in enumerate(self.modules))
        )
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
        logger.debug(f"Generating futures for dataset {dataset_name}")
        files = dsprep.getCoffeaDataset()["files"]
        maybe_base_form = dsprep.coffea_dataset_split.get("form", None)
        if maybe_base_form is not None:
            maybe_base_form = ak.forms.from_json(decompress_form(maybe_base_form))
        events, report = NanoEventsFactory.from_root(
            files,
            schemaclass=NanoAODSchema,
            uproot_options=dict(
                allow_read_errors_with_report=True,
            ),
            known_base_form=maybe_base_form,
            persistent_cache=self.cache,
        ).events()
        daskres = DatasetDaskRunResult(dsprep, {}, ak.num(events, axis=0), report)
        dataset_analyzer = DatasetProcessor(daskres, dsprep.dataset_input.fill_name)
        pr.enable()
        for m in self.modules:
            logger.info(f"Adding module {m.name} to dataset {dataset_name}")
            test = m(events, dataset_analyzer)
            events, dataset_analyzer = test
        pr.disable()
        return daskres

    def execute(self, futures: Iterable[DatasetDaskRunResult], client: Client):
        futures = list(futures)
        logger.debug(f"Executing {len(futures)} analysis futures.")
        dsk = {
            x.getName(): [
                x.dataset_preprocessed,
                x.histograms,
                x.raw_events_processed,
                x.run_report,
            ]
            for x in futures
        }

        if client is None:
            computed, *rest = dask.compute(dsk, scheduler="single-threaded")
        else:
            f = client.compute(dsk)
            computed = client.gather(f)

        return {
            name: DatasetRunResult(prep, h, r, rep)
            for name, (prep, h, r, rep) in computed.items()
        }


@dataclass
class ResultModification:
    user: str
    time: datetime


def mergeAndWeightResults(results, sample_manager, target_lumi=None):
    return utils.accumulate(
        [x.getScaledHistograms(sample_manager, target_lumi) for x in results]
    )


@dataclass
class AnalysisResult:
    # modifications: List[ResultModification]
    results: Dict[str, DatasetRunResult]

    def save(self, output_file):
        path = Path(output_file)
        parent = path.parent
        parent.mkdir(exist_ok=True, parents=True)
        with open(path, "wb") as f:
            pkl.dump(self, f)

    @staticmethod
    def fromFile(path):
        path = Path(path)
        with open(path, "rb") as f:
            ret = pkl.load(f)
        if not isinstance(ret, AnalysisResult):
            raise RuntimeError(f"File {path} does not contain an analysis result")
        return ret

    def getMergedHistograms(self, sample_manager, target_lumi=None):
        return mergeAndWeightResults(self.results.values(), sample_manager, target_lumi)


@dataclass
class AnalysisInspectionResult:
    type: str
    passed: bool
    description: str


class NEventChecker:
    def __init__(self, sample_manager):
        self.sample_manager = sample_manager

    def __call__(self, result):
        expected = self.sample_manager.getSet(result.getName()).n_events
        actual = result.raw_events_processed
        if expected == actual:
            return AnalysisInspectionResult(
                "Number Events",
                True,
                f"Expected {expected}, found {expected}",
            )
        else:
            return AnalysisInspectionResult(
                "Number Events",
                True,
                f"Expected {expected}, found {expected}",
            )


class InputChecker:
    def __init__(self, sample_manager):
        self.sample_manager = sample_manager

    def __call__(self, result):
        sample = self.sample_manager.getSet(result.getName())
        files = set(parse.urlparse(x.getFile())[2] for x in sample.files)
        prepped = result.dataset_preprocessed
        cof_dataset = prepped.coffea_dataset_split
        cof_files = set(
            parse.urlparse(x)[2] for x in cof_dataset["files"].keys()
        )
        diff = files.difference(cof_files)
        if diff:
            return AnalysisInspectionResult(
                "Input Files",
                False,
                f"Missing files from input {diff} from analysis input",
            )
        else:
            return AnalysisInspectionResult(
                "Input Files",
                True,
                f"All files in sample found in input to analyzer",
            )


def checkDatasetResult(ds_result, sample_manager):
    checkers = [NEventChecker(sample_manager), InputChecker(sample_manager)]
    results = [checker(ds_result) for checker in checkers]
    return results


def checkAnalysisResult(result, sample_manager):
    ret = {
        name: checkDatasetResult(ds_res, sample_manager)
        for name, ds_res in result.results.items()
    }
    return ret
