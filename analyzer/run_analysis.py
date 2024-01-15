import analyzer.modules
from analyzer.datasets import loadSamplesFromDirectory
from .histogram_builder import HistogramBuilder
import analyzer.core as core
import coffea.dataset_tools as dst
from coffea.dataset_tools.apply_processor import DaskOutputType
from coffea.dataset_tools.preprocess import DatasetSpec
import dask
import hist.dask as dah
import hist

from typing import Any, Callable, Dict, Hashable, List, Set, Tuple, Union, Optional
import dask_awkward as dak
import sys
import shutil
import itertools as it
import awkward as ak

from coffea.nanoevents import BaseSchema, NanoAODSchema, NanoEventsFactory
from coffea.analysis_tools import PackedSelection, Weights

import pickle

from pathlib import Path

from dataclasses import dataclass


class AnalyzerError(Exception):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class DatasetAnalyzer:
    def __init__(self, analyzer, sample_name, fill_name, dataset_weight):
        self.analyzer = analyzer
        self.sample_name = sample_name
        self.fill_name = fill_name
        self.dataset_weight = dataset_weight

        self.histogram_builder = None

        self.report = None
        self.hmaker = None

    @property
    def selection(self):
        if self.sample_name not in self.analyzer._selections:
            self.analyzer._selections[self.sample_name] = PackedSelection()
        return self.analyzer._selections[self.sample_name]

    def applySelection(self, events):
        events = events[self.selection.all(*self.selection.names)]
        self.prepHistogramBuilder(events)
        print(f"Current selection is {self.selection}")
        return events

    def prepHistogramBuilder(self, events):
        self.histogram_builder.setEventWeights(events.EventWeight)

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
        if key in self.analyzer._dask_histograms:
            self.histogram_builder.fillHistogram(
                self.analyzer._dask_histograms[key], data, mask
            )
        else:
            self.analyzer._dask_histograms[
                key
            ] = self.histogram_builder.createHistogram(axis, data, name, description)
            self.histogram_builder.fillHistogram(
                self.analyzer._dask_histograms[key], data, mask
            )

    def H(self, *args, **kwargs):
        return self.maybeCreateAndFill(*args, **kwargs)


@dataclass
class DatasetInput:
    dataset_name: str
    fill_name: str
    dataset_weight: float
    coffea_dataset: DatasetSpec
    target_lumi: Optional[float]

    def getDatasetWeight(self):
        pass

    @staticmethod
    def fromSampleOrCollection(sample_or_collection, target_lumi=None):
        return [
            DatasetInput(
                s.name, s.fillname, s.dataset_weight, s.coffea_dataset, target_lumi
            )
            for s in sample_or_collection.getAnalyzerSamples(target_lumi)
        ]


@dataclass
class DatasetPreprocessed:
    dataset_input: DatasetInput
    coffea_dataset_split: DatasetSpec

    @staticmethod
    def fromDatasetInput(dataset_input, client, **kwargs):
        out, x = dst.preprocess(dataset_input.coffea_dataset, **kwargs)
        return DatasetPreprocessed(dataset_input, out)


@dataclass
class DatasetRunState:
    dataset_preprocessed: DatasetPreprocessed
    raw_events_processed: int = 0
    dataset_run_report: Optional[ak.Array] = None

    def getCoffeaDataset(self) -> DatasetSpec:
        return self.dataset_preprocessed.coffea_dataset_split[
            self.input_dataset.dataset_name
        ]

    @property
    def input_dataset(self):
        return self.dataset_preprocessed.dataset_input


@dataclass
class DatasetDaskRunState:
    dataset_preprocessed: DatasetPreprocessed
    raw_events_processed: Any
    dataset_run_report: dak.Array


@dataclass
class AnalyzerDaskResult:
    histograms: Dict[str, dah.Hist]
    dataset_run_states: Dict[str, DatasetDaskRunState]


@dataclass
class AnalyzerResult:
    histograms: Dict[str, hist.Hist]
    dataset_run_states: Dict[str, DatasetRunState]


class Analyzer:
    """
    Represents an analysis, a collection of modules.
    """

    def __init__(self, modules, cache):
        self.modules: List[core.AnalyzerModule] = self.__createAndSortModules(*modules)
        self.cache = cache

        self._selections: Dict[str, PackedSelection] = {}
        self._dask_histograms: Dict[str, dah.Hist] = {}
        self._dataset_states: Dict[str, DatasetDaskRunState] = {}

    def __createAndSortModules(self, *module_names):
        m = core.namesToModules(module_names)
        t = core.generateTopology(m)
        modules = core.sortModules(m)
        return modules

    def registerToCompute(self, dataset_states: List[DatasetRunState]):
        for dataset_state in dataset_states:
            dataset_name = dataset_state.input_dataset.dataset_name
            files = dataset_state.getCoffeaDataset()["files"]
            events, report = NanoEventsFactory.from_root(
                files,
                schemaclass=NanoAODSchema,
                uproot_options={"allow_read_errors_with_report": True},
                persistent_cache=self.cache,
            ).events()
            dataset_analyzer = DatasetAnalyzer(
                self,
                dataset_name,
                dataset_state.input_dataset.fill_name,
                dataset_state.input_dataset.dataset_weight,
            )
            if "genWeight" in events.fields:
                events["EventWeight"] = (
                    dak.where(events.genWeight > 0, 1, -1)
                    * dataset_analyzer.dataset_weight
                )
            else:
                events["EventWeight"] = dataset_analyzer.dataset_weight

            dataset_analyzer.histogram_builder = HistogramBuilder(events.EventWeight)
            for m in self.modules:
                print(f"Adding module {m}")
                #events = m(events, dataset_analyzer)
                events = m(events, dataset_analyzer)
            self._dataset_states[dataset_name] = DatasetDaskRunState(
                dataset_state.dataset_preprocessed, dak.num(events, axis=0), report
            )

    def execute(self, client) -> AnalyzerResult:
        hists, states = client.compute(
            [
                self._dask_histograms,
                {
                    x: [y.raw_events_processed, y.dataset_run_report]
                    for x, y in self._dataset_states.items()
                },
            ]
        )

        hists, states = client.gather([hists, states])
        states = {
            x: DatasetRunState(self._dataset_states[x].dataset_preprocessed, *y)
            for x, y in states.items()
        }
        return hists, states


def createLPCCondorCluster(configuration):
    from distributed import Client
    from lpcjobqueue import LPCCondorCluster
    import os

    logpath = Path("/uscmst1b_scratch/lpc1/3DayLifetime/") / os.getlogin() / "dask_logs"
    logpath.mkdir(exist_ok=True, parents=True)
    cluster = LPCCondorCluster(
        host="tcp://localhost:8787", memory="2.0G", log_directory=logpath
    )
    return cluster


def createLocalCluster(configuration):
    from distributed import Client, TimeoutError, LocalCluster

    local_cluster = LocalCluster(
        "tcp://localhost:8787", timeout="2s", memory_limit="4.0G"
    )
    return local_cluster


cluster_factory = dict(
    local=createLocalCluster,
    lpccondor=createLPCCondorCluster,
)


# def runAnalysis():
#    args = getArguments()
#
#    md = {}
#    if args.metadata_cache:
#        md_path = Path(args.metadata_cache)
#        if md_path.is_file():
#            print(f"Loading metadata from {md_path}")
#            md = pickle.load(open(md_path, "rb"))
#
#    runner = createRunner(
#        executor=args.executor,
#        parallelism=args.parallelism,
#        chunk_size=args.chunk_size,
#        max_chunks=args.max_chunks,
#        metadata_cache=md,
#    )
#
#    sample_manager = loadSamplesFromDirectory(args.dataset_dir)
#    all_samples = sample_manager.possibleInputs()
#
#    exist_path = args.update_existing or args.check_file
#
#    existing_data = None
#    if exist_path:
#        update_file_path = Path(exist_path)
#        with open(update_file_path, "rb") as f:
#            existing_data = pickle.load(f)
#
#    if args.check_file:
#        check(existing_data, sample_manager, runner)
#        sys.exit(0)
#
#    list_mode = False
#    if args.list_samples:
#        list_mode = True
#        for x in all_samples:
#            print(x)
#    if args.list_modules:
#        list_mode = True
#        for x in all_modules:
#            print(x)
#    if list_mode:
#        sys.exit(0)
#
#    for sample in args.samples:
#        if sample not in all_samples:
#            print(
#                f"Sample {sample} is not known, please use --list-samples to show available samples."
#            )
#            sys.exit(1)
#
#    if not (args.samples):
#        print("Error: When not in list mode you must provide samples")
#        sys.exit(1)
#
#    modules = args.module_chain
#    if args.exclude_modules:
#        modules = [x for x in all_modules if x not in args.exclude_modules]
#
#    max_retries = args.max_retries
#    retry_count = 0
#    samples = list(sample_manager[x] for x in args.samples)
#    print(
#        f"Running on the following samples:\n\t- "
#        + "\n\t- ".join(x.name for x in samples)
#    )
#    while retry_count < max_retries:
#        out = workFunction(runner, samples, modules, existing_data, args.target_lumi)
#        existing_data = out
#        check_res = check(out, sample_manager, runner)
#        if check_res:
#            print("All checks passed")
#            break
#        else:
#            print("Not all checks passed, attempting retry")
#            retry_count += 1
#            if retry_count == max_retries:
#                print(
#                    "Checks failed but have reached max retries. You will likely need to rerun the analyzer"
#                )
#
#    if args.metadata_cache:
#        md_path = Path(args.metadata_cache)
#        pickle.dump(md, open(md_path, "wb"))
#
#    if args.output:
#        print(f"Saving output {args.output}")
#        outdir = args.output.parent
#        outdir.mkdir(exist_ok=True, parents=True)
#        pickle.dump(out, open(args.output, "wb"))
