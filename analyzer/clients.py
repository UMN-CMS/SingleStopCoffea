from distributed import Client, TimeoutError, LocalCluster
from lpcjobqueue import LPCCondorCluster
import os
from pathlib import Path
import yaml
import multiprocessing
import time
import analyzer.resources
import importlib.resources
import dask
import atexit
import logging

logger = logging.getLogger(__name__)


def createLPCCondorCluster(configuration):
    workers = configuration["n_workers"]
    memory = configuration["memory"]
    schedd_host = configuration["schedd_host"]
    dash_host = configuration["dashboard_host"]
    logpath = Path("/uscmst1b_scratch/lpc1/3DayLifetime/") / os.getlogin() / "dask_logs"
    logpath.mkdir(exist_ok=True, parents=True)

    cluster = LPCCondorCluster(
        memory=memory,
        ship_env=False,
        image="cmssw/cc7:x86_64",
        transfer_input_files=["setup.sh", "coffeaenv/"],
        log_directory=logpath,
        scheduler_options=dict(
            host=schedd_host,
            dashboard_address=dash_host,
        ),
    )
    cluster.scale(workers)
    print(cluster)
    # cluster.adapt(minimum=4, maximum=workers)
    return cluster


def createLocalCluster(configuration):
    workers = configuration["n_workers"]
    memory = configuration["memory"]
    schedd_host = configuration["schedd_host"]
    dash_host = configuration["dashboard_host"]
    cluster = LocalCluster(
        dashboard_address=dash_host,
        memory_limit=memory,
        n_workers=workers,
        scheduler_kwargs={"host": schedd_host},
    )
    return cluster


cluster_factory = dict(
    local=createLocalCluster,
    lpccondor=createLPCCondorCluster,
)


def createNewCluster(cluster_type, config):
    with importlib.resources.as_file(
        importlib.resources.files(analyzer.resources)
    ) as f:
        dask_cfg_path = Path(f) / "dask_config.yaml"
    with open(dask_cfg_path) as f:
        defaults = yaml.safe_load(f)
    dask.config.update(dask.config.config, defaults, priority="new")
    cluster = cluster_factory[cluster_type](config)
    print(cluster)
    print(cluster.dashboard_link)
    return cluster


def cleanup(p):
    if p.is_alive():
        logger.info("Cleaning up process")
        p.terminate()
        p.join()


def runNewCluster(cluster_type, config):
    # p = multiprocessing.Process(target=createNewCluster, args=(cluster_type, config))
    # atexit.register(cleanup, p)
    # p.start()
    cluster = createNewCluster(cluster_type, config)
    time.sleep(config["timeout"])
    # cleanup(p)


# def runAnalysis():
#    args = getArguments()
#
#    md = {}
#    if args.metadata_cache:
#        md_path = Path(args.metadata_cache)
#        if md_path.is_file():
#            print(f"Loading metadata from {md_path}")
#            md = pickle.load(open(md_path, "rb"))
#
#    runner = createRunner(
#        executor=args.executor,
#        parallelism=args.parallelism,
#        chunk_size=args.chunk_size,
#        max_chunks=args.max_chunks,
#        metadata_cache=md,
#    )
#
#    sample_manager = loadSamplesFromDirectory(args.dataset_dir)
#    all_samples = sample_manager.possibleInputs()
#
#    exist_path = args.update_existing or args.check_file
#
#    existing_data = None
#    if exist_path:
#        update_file_path = Path(exist_path)
#        with open(update_file_path, "rb") as f:
#            existing_data = pickle.load(f)
#
#    if args.check_file:
#        check(existing_data, sample_manager, runner)
#        sys.exit(0)
#
#    list_mode = False
#    if args.list_samples:
#        list_mode = True
#        for x in all_samples:
#            print(x)
#    if args.list_modules:
#        list_mode = True
#        for x in all_modules:
#            print(x)
#    if list_mode:
#        sys.exit(0)
#
#    for sample in args.samples:
#        if sample not in all_samples:
#            print(
#                f"Sample {sample} is not known, please use --list-samples to show available samples."
#            )
#            sys.exit(1)
#
#    if not (args.samples):
#        print("Error: When not in list mode you must provide samples")
#        sys.exit(1)
#
#    modules = args.module_chain
#    if args.exclude_modules:
#        modules = [x for x in all_modules if x not in args.exclude_modules]
#
#    max_retries = args.max_retries
#    retry_count = 0
#    samples = list(sample_manager[x] for x in args.samples)
#    print(
#        f"Running on the following samples:\n\t- "
#        + "\n\t- ".join(x.name for x in samples)
#    )
#    while retry_count < max_retries:
#        out = workFunction(runner, samples, modules, existing_data, args.target_lumi)
#        existing_data = out
#        check_res = check(out, sample_manager, runner)
#        if check_res:
#            print("All checks passed")
#            break
#        else:
#            print("Not all checks passed, attempting retry")
#            retry_count += 1
#            if retry_count == max_retries:
#                print(
#                    "Checks failed but have reached max retries. You will likely need to rerun the analyzer"
#                )
#
#    if args.metadata_cache:
#        md_path = Path(args.metadata_cache)
#        pickle.dump(md, open(md_path, "wb"))
#
#    if args.output:
#        print(f"Saving output {args.output}")
#        outdir = args.output.parent
#        outdir.mkdir(exist_ok=True, parents=True)
#        pickle.dump(out, open(args.output, "wb"))
