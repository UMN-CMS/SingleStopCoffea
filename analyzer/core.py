import enum
from functools import wraps
from collections import namedtuple


class ModuleType(enum.IntEnum):
    PreSelectionProducer = 1
    PreSelectionHist = 2
    MainProducer = 3
    MainHist = 4
    Output = 5


Module = namedtuple("Module", "name type func deps require_tags")
modules = {}


def analyzerModule(name, mod_type, dependencies=None, require_tags=None):
    def decorator(func):
        tags = require_tags if require_tags is not None else []
        modules[name] = Module(name, mod_type, func, dependencies,  set(tags))
        return func
    return decorator
