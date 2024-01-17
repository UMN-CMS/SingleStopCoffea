import analyzer.datasets as ds
import dask
import analyzer.modules
import coffea
import coffea.dataset_tools as dst
import analyzer.core as ac
import itertools as it
from rich import print, inspect
from rich.console import Console
from dask.distributed import Client
from rich.table import Table
import pickle as pkl
import shutil
import tempfile

from analyzer.clients import createLPCCondorCluster, createNewCluster

if __name__ == "__main__":
    #cluster = createNewCluster(
    #    "local",
    #    dict(n_workers=4, memory="2.0G", schedd_host=None, dashboard_host=None),
    #)
    #client = Client(cluster)
    client = Client("localhost:8786")
    print(client)
    # shutil.make_archive("coffeaenv", 'zip', "coffeaenv")
    # client.upload_file("coffeaenv.zip")
    # shutil.make_archive("analyzer", 'zip', "analyzer")
    # client.upload_file("analyzer.zip")
    print(client)
    table = Table()
    table.add_column("Name")
    table.add_column("Categories")
    table.add_column("Dependencies")
    for x in ac.modules.values():
        table.add_row(
            x.name,
            ",".join(x.categories) if x.categories else "",
            ",".join(x.depends_on) if x.depends_on else "",
        )

    console = Console()
    modules = [
        "objects",
        "baseline_selection",
        "weights",
        "dataset_category",
        "event_level",
        "event_level_hists",
        #"jets",
    ]

    cache = {}
    analyzer = ac.Analyzer(modules, cache)

    samples = ds.loadSamplesFromDirectory("datasets")
    wanted = ["signal_312_2000_1900"] #, "signal_312_2000_1400", "Skim_QCDInclusive2018"]
    s = [samples[x] for x in wanted]
    sample2 = samples["signal_312_2000_1900"]
    ss = list(
        it.chain.from_iterable(ac.DatasetInput.fromSampleOrCollection(x) for x in s)
    )

    dataset_preps = [
        ac.DatasetPreprocessed.fromDatasetInput(x, None, maybe_step_size=50000)
        for x in ss
    ]
    # with open("test.pkl", 'wb') as f:
    #    pkl.dump(dataset_prep , f)
    # with open("test.pkl", 'rb') as f:
    #    dataset_prep  = pkl.load(f)

    futures = [analyzer.getDatasetFutures(x) for x in dataset_preps]
    res = analyzer.execute(futures, client)
    print(res)

    with open("resultstest.pkl", "wb") as f:
        pkl.dump(res, f)

    # res, rep = dst.preprocess(
    #    qcd_dataset,
    #    maybe_step_size=100000,
    #    #uproot_options={"allow_read_errors_with_report": True},
    # )
    # print(res)
    # print(rep)
