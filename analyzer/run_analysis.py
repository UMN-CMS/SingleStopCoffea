import argparse
import analyzer.modules
from analyzer.datasets import loadSamplesFromDirectory
from analyzer.core import modules as all_modules
from analyzer.process import AnalysisProcessor
import sys
import shutil

from coffea import processor
from coffea.nanoevents import NanoAODSchema

import pickle

from pathlib import Path


def createDaskCondor(w):
    from distributed import Client
    from lpcjobqueue import LPCCondorCluster

    cluster = LPCCondorCluster()
    cluster.adapt(minimum=10, maximum=w)
    client = Client(cluster)
    shutil.make_archive("analyzer", "zip", base_dir="analyzer")
    client.upload_file("analyzer.zip")
    return processor.DaskExecutor(client=client)


local_cluster = None
def createDaskLocal(w):
    from distributed import Client,TimeoutError, LocalCluster
    global local_cluster
    if local_cluster is None:
        local_cluster = LocalCluster('tcp://localhost:8787', timeout='2s')
    client = Client(local_cluster)
    print(client)
    return processor.DaskExecutor(client=client)


executor_map = dict(
    iterative=lambda w: processor.IterativeExecutor(),
    futures=lambda w: processor.FuturesExecutor(workers=w),
    dask_local=createDaskLocal,
    dask_condor=createDaskCondor,
)

sample_manager = None


def loadSamples(d):
    global sample_manager
    sample_manager = loadSamplesFromDirectory(d)


def runModulesOnSamples(
        modules, samples, executor="iterative", parallelism=8, chunk_size=250000, max_chunks=None
):

    executor = executor_map[executor](parallelism)
    runner = processor.Runner(
        executor=executor, schema=NanoAODSchema, chunksize=chunk_size, skipbadfiles=True, maxchunks=max_chunks
    )
    samples = [sample_manager[sample] for sample in samples]
    tag_sets = iter([s.getTags() for s in samples])
    common_tags = next(tag_sets).intersection(*tag_sets)
    files = {}
    for samp in samples:
        files.update(samp.getFiles())
    wmap = {}
    for samp in samples:
        wmap.update(samp.getWeightMap())

    return runner(
        files,
        "Events",
        processor_instance=AnalysisProcessor(common_tags, modules, wmap),
    )


def runAnalysis():
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
        default="datasets"
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
        default="futures",
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
        default=None
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
    args = parser.parse_args()

    loadSamples(args.dataset_dir)
    all_samples = sample_manager.possibleInputs()

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
            print(f"Sample {sample} is not known, please use --list-samples to show available samples.")
            sys.exit(1)

    if not (args.samples):
        print("Error: When not in list mode you must provide samples")
        sys.exit(1)

    
    modules = args.module_chain
    if args.exclude_modules:
        modules = [x for x in all_modules if x not in args.exclude_modules]

        
    out = runModulesOnSamples(
        modules,
        args.samples,
        executor=args.executor,
        parallelism=args.parallelism,
        chunk_size=args.chunk_size,
        max_chunks=args.max_chunks
    )
    if args.output:
        outdir = args.output.parent
        outdir.mkdir(exist_ok=True, parents=True)
        pickle.dump(out, open(args.output, "wb"))
