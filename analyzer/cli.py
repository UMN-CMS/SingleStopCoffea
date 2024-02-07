import argparse

from pathlib import Path
from rich import print
from rich.console import Console
from rich.table import Table
import sys
import logging

from analyzer.core import modules as all_modules
import analyzer.core as ac
from analyzer.clients import createNewCluster, runNewCluster, cluster_factory
import analyzer.run_analysis as ra
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
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory(args.dataset_path)
    ret = ra.runAnalysisOnSamples(
        args.modules,
        args.samples,
        sample_manager,
        dask_schedd_address=args.scheduler_address,
    )
    print(ret)
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
    import analyzer.modules

    all_modules = list(ac.modules.values())
    table = Table("Name", "Categories", "Depends On", title="Analysis Modules")
    for module in ac.modules.values():
        table.add_row(
            module.name,
            ",".join(x for x in module.categories),
            ",".join(x for x in module.depends_on),
        )
    console = Console()
    console.print(table)


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
    subparser = subparsers.add_parser(
        "modules", help="Get information on available analysis modules"
    )
    subparser.set_defaults(func=handleModules)


def handleCheck(args):
    manager = ds.SampleManager()
    manager.loadSamplesFromDirectory(args.dataset_path)

    print(f"Running checks on file {args.input}")
    res = ac.AnalysisResult.fromFile(args.input)
    checks = ac.checkAnalysisResult(res, manager)

    table = Table(title="Check Result")
    table.add_column("Sample")
    table.add_column("Check Name")
    table.add_column("Passed")
    table.add_column("Info")
    print(checks)
    for sample, sample_checks in checks.items():
        for c in sample_checks:
            table.add_row(sample, c.type, str(c.passed), c.description)

    console = Console()
    console.print(table)


def addSubparserCheck(subparsers):
    subparser = subparsers.add_parser(
        "check-file", help="Identify potential problems in an output file"
    )
    subparser.add_argument("input", type=str, help="Path to the results file to check")
    subparser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path to directory containing dataset files",
    )

    subparser.set_defaults(func=handleCheck)


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

    subparser.add_argument(
        "--dataset-path",
        default="datasets",
        type=str,
        help="Path to directory containing dataset files",
    )

    subparser.add_argument(
        "--save-graph",
        default=None,
        type=str,
        help="In set, interpret as a path to which the task-graph should be saved.",
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
    addSubparserCheck(subparsers)
    addSubparserModules(subparsers)

    args = parser.parse_args()

    if not hasattr(args, "func"):
        parser.print_help(sys.stderr)

    return args
