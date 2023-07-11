from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set, Union
from yaml import load, dump
import itertools as it
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



@dataclass
class SampleFile:
    paths: List[str] = field(default_factory=list)

    @staticmethod
    def fromDict(data):
        if isinstance(data, List):
            return SampleFile(data)
        else:
            return SampleFile([data])

    def getFiles(self):
        return [self.paths[0]]




@dataclass
class SampleSet:
    name: str
    derived_from: str
    produced_on: str
    lumi: float
    x_sec: float
    n_events: int
    files: List[SampleFile] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    @staticmethod
    def fromDict(data):
        name = data["name"]
        derived_from = data["derived_from"]
        produced_on = data["produced_on"]
        lumi = data["lumi"]
        x_sec = data["xsec"]
        n_events = data["n_events"]
        tags = set(data.get("tags", []))
        files = [SampleFile.fromDict(x) for x in data["files"]]

        ss = SampleSet(name,derived_from,produced_on, lumi, x_sec, n_events, files, tags)
        return ss
        

    def getFiles(self):
        return it.chain(f.getFiles() for f in self.files)

    def getWeight(self):
        return lumi * x_sec / n_events

    def getWeightMap(self):
        return {self.name : self.getWeight()}

    def getTags(self):
        return tags

@dataclass
class SampleCollection:
    name: str
    sets: List[SampleSet] = field(default_factory=list)

    @staticmethod
    def fromDict(data, manager):
        name = data["name"]
        sets = data["sets"]
        real_sets = [manager.sets[s] for s in sets]

        sc = SampleCollection(name, real_sets)
        return sc

    def getWeightMap(self):
        merged = {}
        for s in sets:
            merged.update(s.getWeightMap())

    def getTags(self):
        tags = iter(it.chain(s.tags for s in sets))
        ret = next(tags)
        for t in tags:
            ret = t.intersection(ret)
        return ret
        
@dataclass
class SampleManager:
    sets: Dict[str, SampleSet] = field(default_factory=dict)
    collections: Dict[str, SampleCollection] = field(default_factory=dict)

def loadSamplesFromDirectory(directory):
    directory = Path(directory)
    files = directory.glob("*.yaml")
    ret = {}
    manager = SampleManager()
    for f in files:
        with open(f, 'r') as fo:
            data = load(fo, Loader=Loader)
            for d in [x for x in data if x["type"] == "set"]:
                s = SampleSet.fromDict(d)
                manager.sets[s.name] = s
    for f in files:
        with open(f, 'r') as fo:
            data = load(fo, Loader=Loader)
            for d in [x for x in data if x["type"] == "collection"]:
                s = SampleCollection.fromDict(d, manager)
                manager.collections[s.name] = s
    return manager

