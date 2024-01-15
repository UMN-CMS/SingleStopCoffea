import analyzer.datasets as ds
import dask
import analyzer.modules
import coffea
import coffea.dataset_tools as dst
import analyzer.core as ac
import analyzer.run_analysis as ra
import itertools as it


from rich import print, inspect
from rich.console import Console
from dask.distributed import Client
from rich.table import Table
import pickle as pkl


if __name__ == "__main__":
    dask.config.set(num_workers=4)
    client = Client("127.0.0.1:8786")
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
    console.print(table)
    modules = [
        "objects",
        "baseline_selection",
        "dataset_category",
        "event_level",
        "jets",
    ]

    samples = ds.loadSamplesFromDirectory("datasets")
    wanted = ["signal_312_2000_1900", "signal_312_2000_1400"]#, "Skim_QCDInclusive2018"]
    s = [samples[x] for x in wanted]
    sample2 = samples["signal_312_2000_1900"]
    ss = list(
        it.chain.from_iterable(ra.DatasetInput.fromSampleOrCollection(x) for x in s)
    )
    print(ss)

    dataset_preps = [
        ra.DatasetPreprocessed.fromDatasetInput(x, None, maybe_step_size=150000)
        for x in ss
    ]
    # with open("test.pkl", 'wb') as f:
    #    pkl.dump(dataset_prep , f)
    # with open("test.pkl", 'rb') as f:
    #    dataset_prep  = pkl.load(f)

    drs = [ra.DatasetRunState(x) for x in dataset_preps]
    print(drs)
    cache = {}
    analyzer = ra.Analyzer(modules, cache)

    analyzer.registerToCompute(drs)
    res = analyzer.execute(client)
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
