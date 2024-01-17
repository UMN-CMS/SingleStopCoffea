import argparse
from analyzer.core import modules as all_modules
from analyzer.clients import createNewCluster, runNewCluster, cluster_factory


def handleCluster(args):
    config = {
        "n_workers": args.workers,
        "memory": args.memory,
        "schedd_host": args.scheduler,
        "dashboard_host": args.dashboard,
        "timeout": args.timeout,
    }
    client = runNewCluster(args.type, config)


def addSubparserSamples(subparsers):
    pass


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
    subparser.add_argument("-o", "--output", type=Path, help="Output data path.")
    subparser.add_argument(
        "--skimpath", default=None, type=str, help="Output file for skims"
    )
    subparser.add_argument(
        "-c",
        "--cluster",
        type=str,
        help="Type of cluster to use",
        choices=list(cluster_factory),
        metavar="",
        default="dask_local",
    )
    subparser.add_argument(
        "-t",
        "--metadata-cache",
        type=Path,
        help="File to store and load metadata from",
        default=None,
    )
    subparser.add_argument(
        "-m",
        "--modules",
        type=str,
        nargs="*",
        help="Modules to execute",
        metavar="",
    )

    subparser.add_argument(
        "-l",
        "--target-lumi",
        type=float,
        help="Target luminosity, all samples will have their luminosity scaled to match the target",
        default=None,
    )


def runCli():
    parser = argparse.ArgumentParser(prog="SingleStopAnalyzer")
    subparsers = parser.add_subparsers()
    addSubparserCluster(subparsers)
    args = parser.parse_args()
    args.func(args)
    return args
