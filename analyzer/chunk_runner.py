from coffea import processor
from coffea.processor.executor import (
    WorkItem,
    ProcessorABC,
    DaskExecutor,
    WorkQueueExecutor,
)

from functools import partial

from collections.abc import Mapping, MutableMapping
from typing import (
    Awaitable,
    Callable,
    Dict,
    Generator,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)

def runChunks(self, chunks, processor_instance, treename):
    import lz4.frame as lz4f
    import cloudpickle
    if self.processor_compression is None:
        pi_to_send = processor_instance
    else:
        pi_to_send = lz4f.compress(
            cloudpickle.dumps(processor_instance),
            compression_level=self.processor_compression,
        )
    # hack around dask/dask#5503 which is really a silly request but here we are
    if isinstance(self.executor, DaskExecutor):
        self.executor.heavy_input = pi_to_send
        closure = partial(
            self._work_function,
            self.format,
            self.xrootdtimeout,
            self.mmap,
            self.schema,
            partial(self.get_cache, self.cachestrategy),
            self.use_dataframes,
            self.savemetrics,
            processor_instance="heavy",
        )
    else:
        closure = partial(
            self._work_function,
            self.format,
            self.xrootdtimeout,
            self.mmap,
            self.schema,
            partial(self.get_cache, self.cachestrategy),
            self.use_dataframes,
            self.savemetrics,
            processor_instance=pi_to_send,
        )

    if self.format == "root" and isinstance(self.executor, WorkQueueExecutor):
        # keep chunks in generator, use a copy to count number of events
        # this is cheap, as we are reading from the cache
        chunks_to_count = self.preprocess(fileset, treename)
    else:
        # materialize chunks to list, then count that list
        chunks = list(chunks)
        chunks_to_count = chunks

    events_total = sum(len(c) for c in chunks_to_count)

    exe_args = {
        "unit": "chunk",
        "function_name": type(processor_instance).__name__,
    }
    if isinstance(self.executor, WorkQueueExecutor):
        exe_args.update(
            {
                "unit": "event",
                "events_total": events_total,
                "dynamic_chunksize": self.dynamic_chunksize,
                "chunksize": self.chunksize,
            }
        )

    closure = partial(self.automatic_retries, self.retries, self.skipbadfiles, closure)

    executor = self.executor.copy(**exe_args)
    wrapped_out, e = executor(chunks, closure, None)
    if wrapped_out is None:
        raise ValueError(
            "No chunks returned results, verify ``processor`` instance structure.\n\
            if you used skipbadfiles=True, it is possible all your files are bad."
        )
    wrapped_out["exception"] = e

    if not self.use_dataframes:
        processor_instance.postprocess(wrapped_out["out"])

    if "metrics" in wrapped_out.keys():
        wrapped_out["metrics"]["chunks"] = len(chunks)
        for k, v in wrapped_out["metrics"].items():
            if isinstance(v, set):
                wrapped_out["metrics"][k] = list(v)

    if self.use_dataframes:
        return wrapped_out  # not wrapped anymore
    if self.savemetrics:
        return wrapped_out["out"], wrapped_out["metrics"]
    return wrapped_out["out"]


processor.Runner.runChunks = runChunks
