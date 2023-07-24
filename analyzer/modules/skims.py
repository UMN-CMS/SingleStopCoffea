import analyzer.file_utils as futil
from analyzer.skim import uprootWriteable
import uproot

#@analyzerModule("save_skim", ModuleType.Output)

def saveSkim(events, path):
    print(events.metadata)
    num_events = ak.size(events, axis=0)
    print(num_events)
    if not num_events:
        print("No Events, not saving skimmed file")
        return ""
    scratch_dir_name = os.environ.get("ANALYSIS_SCRATCH_DIR", "./scratch")
    estart, estop = events.metadata["entrystart"], events.metadata["entrystop"]
    filename=f"{futil.getStem(events.metadata['filename'])}_{estart}_{estop}.root"
    scratch_path = Path(scratch_dir_name)
    if not scratch_path.is_dir():
        scratch_path.mkdir()
    temppath = scratch_path / filename
    with uproot.recreate(temppath) as fout:
        writeable = uprootWriteable(events,fields=["Jet", "FatJet", "EventWeight", "Electron", "Muon", "MET", "GenPart"])
        fout["Events"] = writeable
    outpath = futil.appendToUrl(path, events.metadata["dataset"], filename)
    futil.copyFile(temppath,  outpath)
    return outpath
