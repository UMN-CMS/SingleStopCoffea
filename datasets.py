from pathlib import Path
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import accumulate, ProcessorABC
from coffea.processor import FuturesExecutor
import concurrent.futures 
from functools import wraps,partial
from dataclasses import dataclass








fbase = Path("samples")
samples = [
        "QCD2018",
        "Diboson2018",
        "WQQ2018",
        "ZQQ2018",
        "ST2018",
        "ZNuNu2018",
        "TT2018",
        "signal_1000_400",
        "signal_1000_600",
        "signal_1000_900",
        "signal_1500_1400",
        "signal_1500_400",
        "signal_1500_600",
        "signal_1500_900",
        "signal_2000_1400",
        "signal_2000_1900",
        "signal_2000_400",
        "signal_2000_600",
        "signal_2000_900"]

filesets = {
        sample: [
            f"root://cmsxrootd.fnal.gov//store/user/ckapsiak/SingleStop/Skims/Skim_2023_06_20/{sample}.root"
            ]
        for sample in samples
        #if "Di" in sample
        }


def getEvents(datasets, samples):
    ret  =  {s :  NanoEventsFactory.from_root(
        filesets[s][0],
        schemaclass=NanoAODSchema.v6,
        metadata={"dataset": s},
        ).events() for s in samples}
    return ret


def makeClass(name, process_func, Base):
    def __init__(self):
        Base.__init__(self)
    newclass = type(name, (Base,),{"__init__": __init__, "process" : process_function})
    return newclass


def pack(f,i):
    return f(*i)

def runOverDatasets(func, dataset):

    d = [tuple(x) for x in dataset.items()]
    ret = {}
    elems = [x[1] for x in d]
    fe = FuturesExecutor(workers=4)

    ret = fe(d,partial(pack,func),  accumulate)
    return ret
