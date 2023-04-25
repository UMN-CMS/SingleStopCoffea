import importlib
from pathlib import Path
from coffea import processor
from coffea.util import save
import sys
import shutil


def getLocalExecutor(config):
    executor_name = config["executor"]
    if "serial" in executor_name:
        return processor.IterativeExecutor()
    elif "parallel" in executor_name:
        return processor.FuturesExecutor(workers=config["workers"])
    else:
        raise KeyError()
    

def createLocalRunner(config):
    exec_config = config["execution"]
    runner = processor.Runner(
        executor=getLocalExecutor(exec_config),
        schema = config["schema"],
        skipbadfiles=exec_config["skipbadfiles"],
        chunksize=exec_config["chunksize"],
        maxchunks=exec_config["maxchunks"]
    )
    return runner
    


def getDaskExecutor(config):
    from dask.distributed import Client
    if "lpc" in config["execution"]["executor"]:
        from lpcjobqueue import LPCCondorCluster
        cluster = LPCCondorCluster(
            ship_env=True,
            transfer_input_files=[str(Path(x).absolute()) for x in  config["execution"]["transfer_input_files"]],
            log_directory=config["log_directory"],
            memory="2GB")
        cluster.adapt(minimum=config["execution"]["worker_min"], maximum=config["execution"]["worker_max"])
    else:
        raise KeyError()
    client = Client(cluster)
    return processor.DaskExecutor(client=client)

def createDaskRunner(config):
    from dask.distributed import Client
    import dask
    exec_config = config["execution"]
    runner = processor.Runner(
        executor=getDaskExecutor(config),
        schema = config["schema"],
        skipbadfiles=exec_config["skipbadfiles"],
        chunksize=exec_config["chunksize"],
        maxchunks=exec_config["maxchunks"]
    )
    return runner


def executeConfiguration(config):
    executor_name = config["execution"]["executor"]
    log_dir = Path(config["log_directory"])
    shutil.rmtree(log_dir,ignore_errors=True )
    if 'dask' in executor_name:
        runner = createDaskRunner(config)
    elif 'local' in executor_name:
        runner = createLocalRunner(config)
    out = runner(config["datasets"], "Events", processor_instance=config["processor"])
    save(out, config["data_out"])

    
if __name__ == "__main__":
    print(sys.argv[1])
    configuration_file=Path(sys.argv[1])
    spec = importlib.util.spec_from_file_location("config", configuration_file)
    configuration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(configuration)
    config=configuration.config
    print(config)
    executor_name = config["execution"]["executor"]
    if 'dask' in executor_name:
        runner = createDaskRunner(config)
    elif 'local' in executor_name:
        runner = createLocalRunner(config)
    out = runner(config["datasets"], "Events", processor_instance=config["processor"])
    save(out, config["data_out"])


