from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Set
from yaml import load, dump
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

    def getFile(self):
        return self.paths[0]


@dataclass
class SampleSet:
    name: str
    derived_from: str
    produced_on: str
    files: List[SampleFile] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)

    @staticmethod
    def fromDict(data):
        name = data["name"]
        derived_from = data["derived_from"]
        produced_on = data["produced_on"]
        tags = set(data.get("tags", []))
        files = [SampleFile.fromDict(x) for x in data["files"]]
        ss = SampleSet(name,derived_from,produced_on,files, tags)
        return ss
        

    def getFiles(self):
        return [f.getFile() for f in self.files]


def loadSampleSetsFromDirectory(directory):
    directory = Path(directory)
    files = directory.glob("*.yaml")
    ret = {}
    for f in files:
        with open(f, 'r') as fo:
            data = load(fo, Loader=Loader)
            for d in data:
                s = SampleSet.fromDict(d)
                ret[s.name] = s
    return ret

