import analyzer.core as ac
import itertools as it
from rich.console import Console
import analyzer.datasets as ds
from dask.distributed import Client
import dask
import importlib.resources as ir
from pathlib import Path
import tempfile
import shutil
import analyzer
from dask.diagnostics import ProgressBar
import logging


logger = logging.getLogger(__name__)


def createPackageArchive(zip_path=None, archive_type="zip"):
    logger.info("Creating analyzer archive")
    if not zip_path:
        temp_path = Path(tempfile.gettempdir())
    else:
        temp_path = Path(zip_path)
    analyzer_path = Path(ir.files(analyzer))
    trimmed_path = temp_path / "trimmedanalyzer" / "analyzer"
    if trimmed_path.is_dir():
        shutil.rmtree(trimmed_path)
    temp_analyzer = shutil.copytree(
        analyzer_path,
        trimmed_path,
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*~"),
    )
    package_path = shutil.make_archive(
        temp_path / "analyzer",
        archive_type,
        root_dir=trimmed_path.parent,
        base_dir="analyzer",
    )
    final_path = temp_path / f"analyzer.{archive_type}"
    logger.info(f"Created analyzer archive at {final_path}")
    return final_path


def transferAnalyzerToClient(client):
    p = str(createPackageArchive(archive_type="zip"))
    logger.info(f"Transfer file {p} to workers.")
    client.upload_file(p)


def runAnalysisOnSamples(
    modules,
    samples,
    dask_schedd_address=None,
    dataset_directory="datasets",
    step_size=25000,
):
    import analyzer.modules

    cache = {}
    if dask_schedd_address:
        logger.info(f"Connecting client to scheduler at {dask_schedd_address}")
        client = Client(dask_schedd_address)
    else:
        logger.info("No scheduler address provided, running locally")
    sample_manager = ds.SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")
    logger.debug(f"Creating analyzer using {len(modules)} modules")
    analyzer = ac.Analyzer(modules, cache)
    samples = [sample_manager[x] for x in samples]
    all_sets = list(
        it.chain.from_iterable(
            ac.DatasetInput.fromSampleOrCollection(x) for x in samples
        )
    )
    dataset_preps = [
        ac.DatasetPreprocessed.fromDatasetInput(x, None, maybe_step_size=step_size)
        for x in all_sets
    ]
    logger.info(f"Preprocessed data in to {len(dataset_preps)} set")
    futures = [analyzer.getDatasetFutures(x) for x in dataset_preps]
    logger.info(f"Generated {len(futures)} analysis futures")
    with ProgressBar():
        if not dask_schedd_address:
            ret = dask.compute(futures, scheduler="single-threaded")
        else:
            ret = analyzer.execute(futures, client)
    return ret
