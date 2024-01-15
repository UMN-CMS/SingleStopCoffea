from pathlib import Path
from dataclasses import dataclass, field, fields
from typing import List, Dict, Set, Union, Optional
from yaml import load, dump
import itertools as it
from collections.abc import Mapping
from coffea.dataset_tools.preprocess import DatasetSpec

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


class ForbiddenDataset(Exception):
    pass


@dataclass
class AnalyzerSample:
    name: str
    fillname: str
    dataset_weight: float
    coffea_dataset: DatasetSpec


@dataclass
class Style:
    color: Optional[str]
    alpha: Optional[float]

    @staticmethod
    def fromDict(data):
        color = data.get("color")
        op = data.get("alpha")
        return Style(color, op)

    def toDict(self):
        return dict(
            (field.name, getattr(self, field.name))
            for field in fields(self)
            if getattr(self, field.name) is not None
        )


@dataclass
class SampleFile:
    paths: List[str] = field(default_factory=list)

    @staticmethod
    def fromDict(data):
        if isinstance(data, List):
            return SampleFile(data)
        else:
            return SampleFile([data])

    def getRootDir(self):
        return "Events"

    def getFile(self):
        return self.paths[0]


@dataclass
class SampleSet:
    name: str
    title: str
    derived_from: Optional[Union[str, "SampleSet"]]
    produced_on: Optional[str]
    lumi: Optional[float]
    x_sec: Optional[float]
    n_events: int
    files: List[SampleFile] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    style: Optional[Union[Style, str]] = None
    isdata: bool = False
    forbid: Optional[bool] = False

    @staticmethod
    def fromDict(data):
        name = data["name"]
        derived_from = data.get("derived_from", None)
        produced_on = data.get("produced_on", None)
        lumi = data.get("lumi", None)
        x_sec = data.get("x_sec", None)
        n_events = data.get("n_events", None)
        tags = set(data.get("tags", []))
        title = data.get("title", name)
        isdata = data.get("isdata", False)
        forbid = data.get("forbid", None)
        if not (x_sec and n_events and lumi) and not (derived_from or isdata):
            raise Exception(
                f"Every sample must either have a complete weight description, or a derivation. While processing\n {name}"
            )
        files = [SampleFile.fromDict(x) for x in data["files"]]

        style = data.get("style", {})
        if not isinstance(style, str):
            style = Style.fromDict(style)

        ss = SampleSet(
            name,
            title,
            derived_from,
            produced_on,
            lumi,
            x_sec,
            n_events,
            files,
            tags,
            style,
            isdata,
            forbid,
        )
        return ss

    def isForbidden(self):
        if self.forbid is None:
            if self.derived_from is None:
                return False
            else:
                return self.derived_from.isForbidden()
        else:
            return self.forbid

    def getStyle(self):
        return self.style

    def getLumi(self):
        if self.derived_from:
            return self.derived_from.lumi
        else:
            return self.lumi

    def getTitle(self):
        return self.title

    def toCoffeaDataset(self):
        if self.isForbidden():
            raise ForbiddenDataset(
                f"Attempting to access the files for forbidden dataset {self.name}"
            )

        return {self.name: {"files": {f.getFile(): f.getRootDir() for f in self.files}}}

    def getWeight(self, target_lumi=None):
        if self.derived_from:
            w = self.derived_from.getWeight()
        else:
            if self.isdata:
                w = 1
            else:
                w = self.lumi * self.x_sec / self.n_events
        if target_lumi:
            w = w * target_lumi / self.getLumi()
        return w

    def getWeightMap(self, target_lumi=None):
        return {self.name: self.getWeight(target_lumi)}

    def getAnalyzerSamples(self, target_lumi=None):
        return [
            AnalyzerSample(
                name=self.name,
                fillname=self.name,
                coffea_dataset=self.toCoffeaDataset(),
                dataset_weight=self.getWeight(target_lumi),
            )
        ]

    def getTagMap(self):
        return {self.name: set(self.tags)}

    def getTags(self):
        return self.tags

    def totalEvents(self):
        return self.n_events

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


@dataclass
class SampleCollection:
    name: str
    title: str
    sets: List[SampleSet] = field(default_factory=list)
    treat_separate: bool = False
    style: Optional[Union[Style, str]] = None

    @staticmethod
    def fromDict(data, manager, force_separate=False):
        name = data["name"]
        sets = data["sets"]
        title = data.get("title", name)
        real_sets = [manager.sets[s] for s in sets]
        style = data.get("style", {})
        if not isinstance(style, str):
            style = Style.fromDict(style)

        sc = SampleCollection(
            name,
            title,
            real_sets,
            data.get("treat_separate", False or force_separate),
            style,
        )

        return sc

    def getTitle(self):
        return self.title

    def toCoffeaDataset(self):
        everything = {}
        for s in self.sets:
            everything.update(s.toCoffeaDataset())
        if not self.treat_separate:
            everything = {
                f"{self.name}:{name}": files for name, files in everything.items()
            }
        return everything

    def getSampleSets(self):
        return self.sets

    def getAnalyzerSamples(self, target_lumi=None):
        return [
            AnalyzerSample(
                name=x.name,
                fillname=x.name if self.treat_separate else self.name,
                coffea_dataset=x.toCoffeaDataset(),
                dataset_weight=x.getWeight(target_lumi),
            )
            for x in self.getSampleSets()
        ]

    def getWeightMap(self, target_lumi=None):
        merged = {}
        for s in self.sets:
            merged.update(s.getWeightMap(target_lumi))
        return merged

    def getStyle(self):
        return self.style

    def totalEvents(self):
        return sum(s.totalEvents() for s in self.sets)

    def __str__(self):
        return self.name

    def __repr__(self):
        return str(self)


@dataclass
class SampleManager:
    sets: Dict[str, SampleSet] = field(default_factory=dict)
    collections: Dict[str, SampleCollection] = field(default_factory=dict)

    def possibleInputs(self):
        return [*self.sets, *self.collections]

    def __getitem__(self, key):
        if key in self.collections:
            return self.collections[key]
        else:
            return self.sets[key]

    def __iter__(self):
        yield from self.sets
        yield from self.collections


def loadSamplesFromDirectory(directory, force_separate=False):
    directory = Path(directory)
    files = list(directory.glob("*.yaml"))
    ret = {}
    manager = SampleManager()
    for f in files:
        with open(f, "r") as fo:
            data = load(fo, Loader=Loader)
            for d in [x for x in data if x.get("type", "") == "set" or "files" in x]:
                s = SampleSet.fromDict(d)
                manager.sets[s.name] = s
    for s_name in manager.sets:
        derived = manager[s_name].derived_from
        if derived:
            manager.sets[s_name].derived_from = manager.sets[derived]
    for f in files:
        with open(f, "r") as fo:
            data = load(fo, Loader=Loader)
            for d in [
                x for x in data if x.get("type", "") == "collection" or "sets" in x
            ]:
                s = SampleCollection.fromDict(d, manager, force_separate)
                manager.collections[s.name] = s
    for s in manager:
        sample = manager[s]
        style = sample.getStyle()
        if isinstance(style, str):
            manager[s].style = manager[style].getStyle()

    return manager


Dataset = Union[SampleSet, SampleCollection]
