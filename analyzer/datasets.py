from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Union, Optional
from yaml import load, dump
import itertools as it
from collections.abc import Mapping

try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper


@dataclass
class Style:
    color: Optional[str]

    @staticmethod
    def fromDict(data):
        color = data.get("color")
        return Style(color)


@dataclass
class SampleFile:
    paths: List[str] = field(default_factory=list)

    @staticmethod
    def fromDict(data):
        if isinstance(data, List):
            return SampleFile(data)
        else:
            return SampleFile([data])

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
        if not (x_sec and n_events and lumi) and not derived_from:
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
        )
        return ss

    def getStyle(self):
        return self.style

    def getTitle(self):
        return self.title

    def getFiles(self):
        return {self.name: [f.getFile() for f in self.files]}

    def getWeight(self):
        if self.derived_from:
            return self.derived_from.getWeight()
        else:
            return self.lumi * self.x_sec / self.n_events

    def getWeightMap(self):
        return {self.name: self.getWeight()}

    def getTags(self):
        return self.tags

    def getMissing(self, other_files):
        all_files = self.getFiles()
        return {
            name: [f for f in flist if f not in other_files]
            for name, flist in all_files.items()
        }

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

    def getFiles(self):
        everything = {}
        for s in self.sets:
            everything.update(s.getFiles())
        if not self.treat_separate:
            everything = {
                f"{self.name}:{name}": files for name, files in everything.items()
            }
        return everything

    def getWeightMap(self):
        merged = {}
        for s in self.sets:
            merged.update(s.getWeightMap())
        return merged

    def getTags(self):
        tags = iter(it.chain(s.tags for s in self.sets))
        ret = next(tags)
        for t in tags:
            ret = t.intersection(ret)
        return ret

    def getMissing(self, other_files):
        all_files = self.getFiles()
        return {
            name: [f for f in flist if f not in other_files]
            for name, flist in all_files.items()
        }

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
