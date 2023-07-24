import argparse
import analyzer.modules
from analyzer.datasets import loadSamplesFromDirectory
from analyzer.core import modules as all_modules
from analyzer.process import AnalysisProcessor
import sys

from coffea import processor
from coffea.nanoevents import NanoAODSchema

import pickle

from pathlib import Path

if __name__ == "__main__":
    def createDaskCondor(w):
        from distributed import Client
        from lpcjobqueue import LPCCondorCluster
        cluster = LPCCondorCluster()
        cluster.adapt(minimum=10, maximum=w)
        client = Client(cluster)
        shutil.make_archive("analyzer", "zip", base_dir="analyzer")
        client.upload_file("analyzer.zip")
        return processor.DaskExecutor(client=client)

    def createDaskLocal(w):
        from distributed import Client
        client = Client()
        return processor.DaskExecutor(client=client)

    
    executor_map = dict(
        iterative=lambda w: processor.IterativeExecutor(),
        futures=lambda w: processor.FuturesExecutor(workers=w),
        dask_local=createDaskLocal,
        dask_condor=createDaskCondor,
    )

    sample_manager = loadSamplesFromDirectory("datasets")
    all_samples = sample_manager.possibleInputs()


    parser = argparse.ArgumentParser("Run the RPV Analysis")
    parser.add_argument(
        "-s",
        "--samples",
        nargs="+",
        help="Sample names to run over",
        choices=list(all_samples),
        metavar='',
    )
    parser.add_argument(
        "--signal-re",
        type=str,
        help="Regex to determine if running over signals only",
        default="signal.*",
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
    parser.add_argument(
        "-o", "--output", type=Path, help="Output file for data" 
    )
    parser.add_argument(
        "--skimpath", default=None, type=str, help="Output file for skims" 
    )
    parser.add_argument(
        "-e",
        "--executor",
        type=str,
        help="Exectuor to use",
        choices=list(executor_map),
        metavar='',
        default="futures",
    )
    parser.add_argument(
        "-m",
        "--module-chain",
        type=str,
        nargs="*",
        help="Modules to execture",
        metavar='',
        choices=list(all_modules),
        default=list(all_modules),
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

    if not (args.samples):
        print("Error: When not in list mode you must provide samples")
        sys.exit(1)

    executor = executor_map[args.executor](args.parallelism)
    runner = processor.Runner(
        executor=executor, schema=NanoAODSchema, chunksize=args.chunk_size, skipbadfiles=True
    )


    samples = [sample_manager[sample] for sample in args.samples]
    tag_sets = iter([s.getTags() for s in samples])
    common_tags = next(tag_sets).intersection(*tag_sets)
    files = {}
    for samp in samples:
        files.update(samp.getFiles())
    wmap = {}
    for samp in samples:
        wmap.update(samp.getWeightMap())

    print(f"Using tag set:\n {common_tags}")
    out = runner(
        files, "Events", processor_instance=AnalysisProcessor(common_tags, args.module_chain, wmap, outpath=args.skimpath)
    )
    if args.output:
        outdir = args.output.parent
        outdir.mkdir(exist_ok=True, parents=True)
        pickle.dump(out, open(args.output, "wb"))
