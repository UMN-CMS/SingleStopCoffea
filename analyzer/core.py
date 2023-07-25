import enum
from functools import wraps
from collections import namedtuple


class ModuleType(enum.IntEnum):
    BaseObjectDef=1
    PreSelectionProducer = 2
    PreSelectionHist = 3
    Selection=4
    MainProducer = 5
    MainHist = 6
    Output = 7
    Categories = 8


Module = namedtuple("Module", "name type func deps require_tags after")
modules = {}


def analyzerModule(name, mod_type, dependencies=None, require_tags=None, after=None):
    def decorator(func):
        tags = require_tags if require_tags is not None else []
        real_after = after if after is not None else []
        modules[name] = Module(name, mod_type, func, dependencies,  set(tags), set(real_after))
        return func
    return decorator
