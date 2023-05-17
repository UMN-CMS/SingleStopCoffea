from pathlib import Path
from coffea.nanoevents import NanoEventsFactory, NanoAODSchema
from coffea.processor import accumulate
import concurrent.futures 


fbase = Path("samples")
samples = [
        "QCD2018",
        "Diboson2018",
        "WQQ2018",
        "ZQQ2018",
        "ST2018",
        "ZNuNu2018",
        "TT2018",
        "signal_1000_400_Skim",
        "signal_1000_600_Skim",
        "signal_1000_900_Skim",
        "signal_1500_1400_Skim",
        "signal_1500_400_Skim",
        "signal_1500_600_Skim",
        "signal_1500_900_Skim",
        "signal_2000_1400_Skim",
        "signal_2000_1900_Skim",
        "signal_2000_400_Skim",
        "signal_2000_600_Skim",
        "signal_2000_900_Skim"]

filesets = {
        sample: [
            f"root://cmsxrootd.fnal.gov//store/user/ckapsiak/SingleStop/Skims/Skim_2023_05_11/{sample}.root"
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


def runOverDataSets(func, datasets):
    return {s : func(e) for s,e in datasets.items()}

def runAndAccum(func,data):
    return accumulate(func(x) for x in data.values())



