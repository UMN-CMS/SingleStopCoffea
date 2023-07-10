from pathlib import Path
from dataclasses import dataclass
from typing import List, Dict
from yaml import load, dump
try:
    from yaml import CLoader as Loader, CDumper as Dumper
except ImportError:
    from yaml import Loader, Dumper



@dataclass
class SampleFile:
    paths: List[str]

    @staticmethod
    def fromDict(data):
        if isinstance(data, List):
            return SampleFile(data)
        else:
            return SampleFile([data])

    def getFiles(self):
        return paths[0]


@dataclass
class SampleSet:
    name: str
    derived_from: str
    produced_on: 
    files: List[SampleFile]

    @staticmethod
    def fromDict(data):
        name = data["name"]
        derived_from = data["derived_from"]
        produced_on = data["produced_on"]
        files = [SampleFile.fromDict(x) for x in data["files"]]

    def getFiles(self):
        return [f.getFile() for f in files]


def loadSampleSetsFromDirectory(directory):
    directory = Path(directory)
    files = directory.glob("*.yaml")
    ret = {}
    for f in files:
        with open(f, 'r') as fo:
            data = load(fo, Loader=Loader)
            s = SampleSet(data)
            ret[s.name] = s
    return ret

    

