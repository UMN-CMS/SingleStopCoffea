from pathlib import Path
import logging
import uproot
import awkward as ak


def is_rootcompat(a):
    """Is it a flat or 1-d jagged array?"""
    t = ak.type(a)
    if isinstance(t, ak._ext.ArrayType):
        if isinstance(t.type, ak._ext.PrimitiveType):
            return True
        if isinstance(t.type, ak._ext.ListType) and isinstance(
            t.type.type, ak._ext.PrimitiveType
        ):
            return True
    return False


def uproot_writeable(events):
    """Restrict to columns that uproot can write compactly"""
    out = {}
    for bname in events.fields:
        if events[bname].fields:
            out[bname] = ak.zip(
                {
                    n: ak.packed(ak.without_parameters(events[bname][n]))
                    for n in events[bname].fields
                    if is_rootcompat(events[bname][n])
                }
            )
        else:
            out[bname] = ak.packed(ak.without_parameters(events[bname]))
    return out



def copy_file( fname, localdir, location, subdirs):
    a_logger.debug( f"Starting file transfer") 
    subdirs = subdirs or []
    xrd_prefix = "root://"
    pfx_len = len(xrd_prefix)
    xrootd = False
    if xrd_prefix in location:
        try:
            import XRootD
            import XRootD.client

            xrootd = True
        except ImportError as err:
            raise ImportError(
                "Install XRootD python bindings with: conda install -c conda-forge xroot"
            ) from err
    a_logger.debug( f"XRootD is {xrootd}") 
    local_file = (
        os.path.abspath(os.path.join(localdir, fname))
        if xrootd
        else os.path.join(localdir, fname)
    )
    a_logger.debug( f"Local file is {local_file}") 
    merged_subdirs = "/".join(subdirs) if xrootd else os.path.sep.join(subdirs)
    destination = (
        location + merged_subdirs + f"/{fname}"
        if xrootd
        else os.path.join(location, os.path.join(merged_subdirs, fname))
    )
    a_logger.debug( f"Destination is {destination}") 
    if xrootd:
        copyproc = XRootD.client.CopyProcess()
        a_logger.debug( f"Creating copy client") 
        copyproc.add_job(local_file, destination)
        copyproc.prepare()
        a_logger.debug( f"Running copy process") 
        copyproc.run()
        client = XRootD.client.FileSystem(
            location[: location[pfx_len:].find("/") + pfx_len]
        )
        status = client.locate(
            destination[destination[pfx_len:].find("/") + pfx_len + 1 :],
            XRootD.client.flags.OpenFlags.READ,
        )
        a_logger.debug( f"Startus of transfer is {status}") 
        assert status[0].ok
        a_logger.debug( f"Cleaning up") 
        del client
        del copyproc
    else:
        dirname = os.path.dirname(destination)
        if not os.path.exists(dirname):
            pathlib.Path(dirname).mkdir(parents=True, exist_ok=True)
        shutil.copy(local_file, destination)
        assert os.path.isfile(destination)
    a_logger.debug( f"Removing orginal file") 
    pathlib.Path(local_file).unlink()

def save_skim(base_path, scratch_dir_name=None):
    def func(events):
        scratch_dir_name = os.environ.get("ANALYSIS_SCRATCH_DIR", "./scratch")

        a_logger.debug( "Starting skimming process") 

        filename = (
               "__".join(
                   [
                       events.metadata['dataset'],
                       events.metadata['fileuuid'],
                       str(events.metadata['entrystart']),
                       str(events.metadata['entrystop']),
                   ]
               )
               + ".root"
            )
        a_logger.debug( f"Skim filename is {filename}") 
        path = Path(base_path)
        scratch_path = Path(scratch_dir_name)
        if not scratch_path.is_dir():
            scratch_path.mkdir()
        outpath = scratch_path / filename
        a_logger.debug( f"Writing temporary root file to {outpath}") 
        with uproot.recreate(outpath) as fout:
            fout["Events"] = uproot_writeable(events)
        a_logger.debug( f"Copying root file {outpath} to {base_path}") 
        copy_file(filename, scratch_path, base_path, [events.metadata["dataset"]])
    return func
