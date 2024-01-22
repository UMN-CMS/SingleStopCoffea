import argparse
from analyzer.core import modules as all_modules
from analyzer.clients import createNewCluster, runNewCluster, cluster_factory
import analyzer.run_analysis as ra
from pathlib import Path
from rich import print
from rich.console import Console
import sys
import logging
import analyzer.datasets as ds


logger = logging.getLogger(__name__)


def handleCluster(args):
    logger.info("Handling cluster-start")
    config = {
        "n_workers": args.workers,
        "memory": args.memory,
        "schedd_host": args.scheduler,
        "dashboard_host": args.dashboard,
        "timeout": args.timeout,
    }
    client = runNewCluster(args.type, config)


def handleRunAnalysis(args):
    logger.info("Handling run analysis")
    ret = ra.runAnalysisOnSamples(args.modules, args.samples, args.scheduler_address)
    ret.save(args.output)



def handleSamples(args):
    logger.info("Handling sample inspection")
    manager = ds.SampleManager()
    manager.loadSamplesFromDirectory("datasets")
    if args.type is None:
        table = ds.createSampleAndCollectionTable(manager, re_filter=args.filter)
    elif args.type == "set":
        table = ds.createSetTable(manager, re_filter=args.filter)
    elif args.type == "collection":
        table = ds.createCollectionTable(manager, re_filter=args.filter)
    console = Console()
    console.print(table)


def handleModules(args):
    logger.info("Handling module inspection")


def addSubparserSamples(subparsers):
    subparser = subparsers.add_parser(
        "samples", help="Get information on available data samples"
    )
    subparser.set_defaults(func=handleSamples)
    subparser.add_argument("-f", "--filter", default=None, help="Regex to filter names")
    subparser.add_argument(
        "-t",
        "--type",
        choices=["set", "collection"],
        default=None,
        help="Show information for only samples or collections",
    )


def addSubparserModules(subparsers):
    pass


def addSubparserCluster(subparsers):
    subparser = subparsers.add_parser("cluster-start", help="Start analysis cluster")
    subparser.add_argument(
        "-t",
        "--type",
        type=str,
        required=True,
        choices=list(cluster_factory),
        help="Type of cluster",
    )
    subparser.add_argument(
        "-d",
        "--dashboard",
        type=str,
        default="localhost:8787",
        help="Host for scheduler",
    )
    subparser.add_argument(
        "-s",
        "--scheduler",
        type=str,
        default=None,
        help="Host for scheduler",
    )
    subparser.add_argument("-w", "--workers", type=int, help="Number of workers")
    subparser.add_argument(
        "-m", "--memory", default="2.0G", type=str, help="Worker memory"
    )
    subparser.add_argument(
        "--timeout",
        type=int,
        default=3600,
        help="Maximum time in seconds to run the cluster",
    )
    subparser.set_defaults(func=handleCluster)


def addSubparserRun(subparsers):
    subparser = subparsers.add_parser(
        "run", help="Run analyzer over some collection of samples"
    )
    subparser.add_argument(
        "-o", "--output", required=True, type=Path, help="Output data path."
    )

    run_mode = subparser.add_mutually_exclusive_group()
    run_mode.add_argument(
        "-a",
        "--scheduler-address",
        type=str,
        help="Address of the scheduler to use for dask",
    )

    run_mode.add_argument(
        "-y",
        "--run-synchronous",
        help="Do not use dask, instead run synchronously.",
    )

    subparser.add_argument(
        "-s",
        "--samples",
        type=str,
        nargs="+",
        help="List of samples to run over",
        metavar="",
    )
    subparser.add_argument(
        "-m",
        "--modules",
        type=str,
        nargs="+",
        help="List of modules to execute.",
        metavar="",
    )
    subparser.set_defaults(func=handleRunAnalysis)


def addGeneralArguments(parser):
    parser.add_argument("--log-level", type=str, default="WARN", help="Logging level")


def runCli():
    parser = argparse.ArgumentParser(prog="SingleStopAnalyzer")
    addGeneralArguments(parser)

    subparsers = parser.add_subparsers()
    addSubparserCluster(subparsers)
    addSubparserRun(subparsers)
    addSubparserSamples(subparsers)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)
        return None
    return args
