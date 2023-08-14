import argparse
import analyzer.modules
from analyzer.datasets import loadSamplesFromDirectory
from analyzer.core import modules as all_modules
from analyzer.process import AnalysisProcessor
import analyzer.chunk_runner
import sys
import shutil

from coffea import processor
from coffea.processor import accumulate
from coffea.processor.executor import WorkItem
import uuid

from coffea.nanoevents import NanoAODSchema


import pickle

from pathlib import Path
import subprocess


def get_git_revision_hash() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()


def get_git_revision_short_hash() -> str:
    return (
        subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
        .decode("ascii")
        .strip()
    )


def createDaskCondor(w):
    from distributed import Client
    from lpcjobqueue import LPCCondorCluster
    import os

    logpath = Path("/uscmst1b_scratch/lpc1/3DayLifetime/") / os.getlogin() / "dask_logs"
    logpath.mkdir(exist_ok=True, parents=True)
    cluster = LPCCondorCluster(memory="2.0G", log_directory=logpath)
    cluster.adapt(minimum=10, maximum=w)
    client = Client(cluster)
    print(client.get_versions())
    shutil.make_archive("analyzer", "zip", base_dir="analyzer")
    client.upload_file("analyzer.zip")
    return processor.DaskExecutor(client=client)


local_cluster = None


def createDaskLocal(w):
    from distributed import Client, TimeoutError, LocalCluster

    global local_cluster
    if local_cluster is None:
        local_cluster = LocalCluster("tcp://localhost:8787", timeout="2s")
    client = Client(local_cluster)
    print(client)
    return processor.DaskExecutor(client=client)


executor_map = dict(
    iterative=lambda w: processor.IterativeExecutor(),
    futures=lambda w: processor.FuturesExecutor(workers=w),
    dask_local=createDaskLocal,
    dask_condor=createDaskCondor,
)


def sampleToFileList(samples):
    # samples = [sample_manager[sample] for sample in samples]
    files = {}
    for samp in samples:
        files.update(samp.getFiles())
    return files


def createRunner(
    executor="iterative",
    parallelism=8,
    chunk_size=250000,
    max_chunks=None,
    metadata_cache=None,
):

    executor = executor_map[executor](parallelism)
    runner = processor.Runner(
        executor=executor,
        schema=NanoAODSchema,
        chunksize=chunk_size,
        skipbadfiles=True,
        maxchunks=max_chunks,
        metadata_cache=metadata_cache,
        savemetrics=True,
        xrootdtimeout=60,
    )
    return runner


def normalizeChunks(chunks):
    normalized_chunks = [
        WorkItem(
            x.dataset,
            x.filename,
            x.treename,
            x.entrystart,
            x.entrystop,
            str(uuid.UUID(bytes=x.fileuuid)),
        )
        for x in chunks
    ]
    return normalized_chunks


def getChunksFromFiles(flist, runner, retries=3):
    chunks = runner.preprocess(flist, "Events")
    chunks = list(chunks)
    return chunks


def runModulesOnSamples(modules, samples, chunks, runner):
    wmap = {}
    tag_sets = iter([s.getTags() for s in samples])
    common_tags = next(tag_sets).intersection(*tag_sets)
    for samp in samples:
        wmap.update(samp.getWeightMap())

    return runner.runChunks(
        chunks,
        AnalysisProcessor(common_tags, modules, wmap),
        "Events",
    )


def getArguments():
    parser = argparse.ArgumentParser("Run the RPV Analysis")
    parser.add_argument(
        "-s",
        "--samples",
        nargs="+",
        help="Sample names to run over",
        metavar="",
    )
    parser.add_argument(
        "-d",
        "--dataset-dir",
        help="Directory containing data sets",
        type=str,
        default="datasets",
    )
    parser.add_argument(
        "--signal-re",
        type=str,
        help="Regex to determine if running over signals only",
        default="signal.*",
    )
    parser.add_argument(
        "-k",
        "--max-chunks",
        type=int,
        help="Maximum number of chunks",
        default=None,
    )
    parser.add_argument(
        "-r",
        "--max-retries",
        type=int,
        help="Maximum number of retries",
        default=5,
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="List available samples and exit",
    )
    parser.add_argument(
        "--list-modules",
        action="store_true",
        help="List available modules and exit",
    )
    parser.add_argument(
        "--force-separate",
        action="store_true",
        help="Treat sample sets within a collection as separate always.",
    )
    parser.add_argument("-o", "--output", type=Path, help="Output file for data")
    parser.add_argument(
        "--skimpath", default=None, type=str, help="Output file for skims"
    )
    parser.add_argument(
        "-e",
        "--executor",
        type=str,
        help="Exectuor to use",
        choices=list(executor_map),
        metavar="",
        default="dask_local",
    )
    parser.add_argument(
        "-t",
        "--metadata-cache",
        type=Path,
        help="File to store and load metadatafrom",
        default=None,
    )
    parser.add_argument(
        "-m",
        "--module-chain",
        type=str,
        nargs="*",
        help="Modules to execture",
        metavar="",
        choices=list(all_modules),
        default=list(all_modules),
    )
    parser.add_argument(
        "-x",
        "--exclude-modules",
        type=str,
        nargs="*",
        help="Modules to exclude",
        metavar="",
        choices=list(all_modules),
        default=None,
    )
    parser.add_argument(
        "-p",
        "--parallelism",
        type=int,
        help="Level of paralleism to use if running a compatible exectutor",
        default=4,
    )
    parser.add_argument(
        "-c", "--chunk-size", type=int, help="Chunk size to use", default=250000
    )

    parser.add_argument(
        "-u",
        "--update-existing",
        type=str,
        help="Update an existing output by running over skipped files.",
    )

    parser.add_argument(
        "-z", "--check-file", type=str, help="Check an existing output for anomalies"
    )

    args = parser.parse_args()
    return args


def produceChunks(existing_chunks, dataset_info, samples, runner):
    all_files = [x.filename for x in existing_chunks]
    all_wanted_chunks = set(existing_chunks)

    all_missing = {}
    for samp in samples:
        missing = samp.getMissing(all_files)
        all_missing.update(missing)

    any_missing = any(bool(x) for x in all_missing.values())
    if any_missing:
        print(
            "Existing chunk set is not complete, missing chunks for the following files:"
        )
        for k, v in all_missing.items():
            print(f"{k}: ")
            print("\n".join(f"\t- {f}" for f in v))

        missing_chunks = getChunksFromFiles(all_missing, runner)
        all_wanted_chunks += missing_chunks
    already_processed_chunks = set(
        x for y in dataset_info.values() for x in y["work_items"]
    )
    still_needed_chunks = set(all_wanted_chunks).difference(
        set(already_processed_chunks)
    )
    if still_needed_chunks:
        print(
            f"Detected that not all chunks were successfully processed."
            f"Expected {len(all_wanted_chunks)} found that {len(already_processed_chunks)} were succesfully processed"
        )
        print("Must run over the following missing chunks:")
        print("\n".join(f"\t- {f}" for f in still_needed_chunks))
        chunks = still_needed_chunks
    else:
        print(
            "This file appears to be complete, all samples have generated chunks, and all expected chunks have been processed. Nothing left to do."
        )
        return None
    return list(chunks)


def check(data, sample_manager, runner):
    dataset_info = data["dataset_info"]
    existing_chunks = data["all_work_items"]
    existing_samples = list(dataset_info.keys())
    samples = [sample_manager[sample] for sample in existing_samples]
    total_missing = 0
    for samp in samples:
        existing_list = set(x for x in dataset_info[samp.name]["file_data"].keys())
        missing = samp.getMissing(existing_list)
        missing = [x for y in missing.values() for x in y]
        total_missing += len(missing)
        expected_events = samp.totalEvents()
        found_events = sum(
            x["num_raw_events"] for x in dataset_info[samp.name]["file_data"].values()
        )
        if expected_events != found_events:
            print(
                f"Sample {samp} does not have the correct number of events, expected {expected_events}, found {found_events}: missing {expected_events-found_events}"
            )
        else:
            print(
                f"Sample {samp} seems complete, expected {expected_events} events, found {found_events} events"
            )
        if missing:
            print(f"Sample {samp} is missing files:")
            print("\n".join(f"\t-{f}" for f in missing))

    if total_missing > 0:
        print(
            f"Found a total of {total_missing} missing files. You likely want to rerun the analyzer with the --update-existing option to reprocess the failed files"
        )
        return False

    chunks = produceChunks(existing_chunks, dataset_info, samples, runner)
    return not chunks


def workFunction(runner, samples, modules, existing_data=None):
    if existing_data is not None:
        existing_chunks = existing_data["all_work_items"]
        chunks = produceChunks(
            existing_chunks, existing_data["dataset_info"], samples, runner
        )
        chunk_info = {"all_work_items": chunks}
    else:
        flist = sampleToFileList(samples)
        chunks = getChunksFromFiles(flist, runner)
        chunk_info = {"all_work_items": chunks}

    if not chunks:
        print(
            f"Could not produce chunks, this likely means that a file is not accessible through xrd"
        )
        return existing_data

    print(f"Running analyzer over {len(chunks)} chunks")

    try:
        out, metrics = runModulesOnSamples(modules, samples, chunks, runner)
        out["metrics"] = metrics
        out = accumulate([out, chunk_info])
        out["all_work_items"] = list(set(out["all_work_items"]))
        existing_data = existing_data if existing_data is not None else {}
        out = accumulate([existing_data, out])
        return out
    except ValueError as e:
        print(e)
        return existing_data
    return None



def runAnalysis():
    current_git_rev = get_git_revision_hash()
    args = getArguments()

    md = {}
    if args.metadata_cache:
        md_path = Path(args.metadata_cache)
        if md_path.is_file():
            print(f"Loading metadata from {md_path}")
            md = pickle.load(open(md_path, "rb"))

    runner = createRunner(
        executor=args.executor,
        parallelism=args.parallelism,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks,
        metadata_cache=md,
    )

    sample_manager = loadSamplesFromDirectory(args.dataset_dir)
    all_samples = sample_manager.possibleInputs()

    exist_path = args.update_existing or args.check_file

    existing_data = None
    if exist_path:
        update_file_path = Path(exist_path)
        with open(update_file_path, "rb") as f:
            existing_data = pickle.load(f)

    if args.check_file:
        check(existing_data, sample_manager, runner)
        sys.exit(0)

    list_mode = False
    if args.list_samples:
        list_mode = True
        for x in all_samples:
            print(x)
    if args.list_modules:
        list_mode = True
        for x in all_modules:
            print(x)
    if list_mode:
        sys.exit(0)

    for sample in args.samples:
        if sample not in all_samples:
            print(
                f"Sample {sample} is not known, please use --list-samples to show available samples."
            )
            sys.exit(1)

    if not (args.samples):
        print("Error: When not in list mode you must provide samples")
        sys.exit(1)

    modules = args.module_chain
    if args.exclude_modules:
        modules = [x for x in all_modules if x not in args.exclude_modules]

    max_retries = args.max_retries
    retry_count = 0
    while retry_count < max_retries:
        samples = [sample_manager[sample] for sample in args.samples]
        out = workFunction(runner, samples, modules, existing_data)
        existing_data = out
        check_res = check(out, sample_manager, runner)
        if check_res:
            print("All checks passed")
            break
        else:
            print("Not all checks passed, attempting retry")
            retry_count += 1
            if retry_count == max_retries:
                print(
                    "Checks failed but have reached max retries. You will likely need to rerun the analyzer"
                )

    if args.metadata_cache:
        md_path = Path(args.metadata_cache)
        pickle.dump(md, open(md_path, "wb"))

    if args.output:
        print(f"Saving output {args.output}")
        outdir = args.output.parent
        outdir.mkdir(exist_ok=True, parents=True)
        pickle.dump(out, open(args.output, "wb"))
