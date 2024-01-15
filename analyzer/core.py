import enum
from functools import wraps
from collections import namedtuple, defaultdict
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Callable, Set
from graphlib import TopologicalSorter, CycleError
from collections.abc import Iterable
from rich.console import Console, ConsoleOptions, RenderResult
from rich.table import Table
import itertools as it


class AnalyzerGraphError(Exception):
    def __init__(self, message):
        super().__init__(message)


def iterableNotStr(t):
    return isinstance(t, Iterable) and not isinstance(t, str)


def toSet(x):
    if iterableNotStr(x):
        return set(x)
    else:
        return (x,)


class AnalyzerModule:
    def __init__(self, name, function, depends_on=None, category=None):
        self.name = name
        self.function = function
        self.depends_on = toSet(depends_on) if depends_on else set()
        self.categories = toSet(category) if category else set()

    def __call__(self, events, analyzer):
        return self.function(events, analyzer)

    def __str__(self):
        return f"AnalyzerModule({self.name}, depends_on={self.depends_on}, catetories={self.categories})"

    def __repr__(self):
        return str(self)


modules = {}
category_order = {"main": ["selection"]}


def generateTopology(module_list):
    mods = [x.name for x in module_list]
    cats = defaultdict(list)
    for x in module_list:
        for c in x.categories:
            cats[c].append(x.name)
    graph = {}
    for i, module in enumerate(module_list):
        graph[module.name] = module.depends_on
        if i > 0:
            graph[module.name].update(
                {
                    mods[i - 1],
                }
            )
        for m in graph[module.name]:
            if m not in mods:
                raise AnalyzerGraphError(
                    f"Module {module.name} depends on {m}, but was this dependency was not supplied"
                )
    return graph


def namesToModules(module_list):
    return [modules[x] for x in module_list]


def sortModules(module_list):
    graph = generateTopology(module_list)
    try:
        ts = TopologicalSorter(graph)
        ret = tuple(ts.static_order())
    except CycleError as e:
        raise AnalyzerGraphError(
            f"Cyclic dependency detected in module specification:\n {' -> '.join(e.args[1])}\n"
            f"You may need to reorder the modules."
        )
    return namesToModules(ret)


def analyzerModule(name, depends_on=None, categories=None):
    def decorator(func):
        if name in modules:
            raise KeyError(f"A module already exists with the name {name}")
        modules[name] = AnalyzerModule(name, func, depends_on, categories)
        return func

    return decorator
