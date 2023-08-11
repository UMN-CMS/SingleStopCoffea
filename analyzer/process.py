from coffea import processor
from analyzer.core import ModuleType
from analyzer.core import modules as all_modules
from coffea.processor import accumulate
import pickle

import itertools as it
from analyzer.modules.axes import *
import awkward as ak
import warnings

warnings.filterwarnings("ignore", message=r".*Missing cross-reference")
warnings.filterwarnings("ignore", message=r".*In coffea version 0.8")


def makeHistogram(
    axis, dataset, data, weights, name=None, description=None, drop_none=True
):
    if isinstance(axis, list):
        h = hist.Hist(dataset_axis, *axis, storage="weight", name=name)
    else:
        h = hist.Hist(dataset_axis, axis, storage="weight", name=name)
    setattr(h, "description", description)
    if isinstance(axis, list):
        ret = h.fill(dataset, *data, weight=weights)
    else:
        ret = h.fill(dataset, ak.to_numpy(data), weight=weights)
    return ret


def makeCategoryHist(_cat_axes, _cat_vals, event_weights):
    cat_axes = list(_cat_axes)
    cat_vals = list(_cat_vals)

    def internal(
        axis,
        data,
        mask=None,
        name=None,
        description=None,
        auto_expand=True,
    ):
        # print("###############################################")
        # print(f"name: {name}")
        # print(f"axes: {axis}")
        # print(f"Cat axes: {cat_axes}")
        if not isinstance(data, list):
            data = [data]
        if not data:
            raise Exception("No data")
        if isinstance(axis, list):
            all_axes = cat_axes + list(axis)
        else:
            all_axes = cat_axes + [axis]
        h = hist.Hist(*all_axes, storage="weight", name=name)
        setattr(h, "description", description)

        weights = event_weights[mask] if mask is not None else event_weights
        base_category_vals = cat_vals
        if mask is not None:
            base_category_vals = [
                x[mask] if isinstance(x, ak.Array) else x for x in base_category_vals
            ]
        shaped_cat_vals = base_category_vals
        shaped_data_vals = data
        if auto_expand:
            mind, maxd = data[0].layout.minmax_depth
            if maxd > 1:
                ol = ak.ones_like(data[0])
                weights = ak.flatten(ol * weights)
                shaped_cat_vals = [
                    ak.flatten(ol * x) if isinstance(x, ak.Array) else x
                    for x in cat_vals
                ]
                shaped_data_vals = [
                    ak.flatten(x) if isinstance(x, ak.Array) else x
                    for x in shaped_data_vals
                ]
        d = shaped_cat_vals + shaped_data_vals
        # print(f"{name} HIST: {h}")
        # print(f"{name} DATA: {d}")
        # print(f"{name} WEIGHTS: {weights}")
        ret = h.fill(*d, weight=weights)
        return ret

    return internal


def splitChain(chain):
    kfunc = lambda k: k.type
    selected = [all_modules[x] for x in chain]
    selected = sorted(selected, key=lambda x: int(x.type))
    grouped = it.groupby(selected, key=kfunc)
    return grouped


def topologicalSort(source):
    pending = [
        (name, set(deps)) for name, deps in source
    ]  # copy deps so we can modify set in-place
    emitted = []
    while pending:
        next_pending = []
        next_emitted = []
        for entry in pending:
            name, deps = entry
            deps.difference_update(emitted)  # remove deps we emitted last pass
            if deps:  # still has deps? recheck during next pass
                next_pending.append(entry)
            else:  # no more deps? time to emit
                yield name
                emitted.append(
                    name
                )  # <-- not required, but helps preserve original ordering
                next_emitted.append(
                    name
                )  # remember what we emitted for difference_update() in next pass
        if (
            not next_emitted
        ):  # all entries have unmet deps, one of two things is wrong...
            raise ValueError(
                "cyclic or missing dependancy detected: %r" % (next_pending,)
            )
        pending = next_pending
        emitted = next_emitted


class AnalysisProcessor(processor.ProcessorABC):
    def __init__(self, tags, chain, weight_map, outpath=None):
        self.tags = tags
        self.output_path = outpath
        self.signal_only = "signal" in self.tags
        self.weight_map = weight_map
        self.modules = {x: list(y) for x, y in splitChain(chain)}
        self.modules = {
            x: [
                z for z in y if z.require_tags.intersection(self.tags) == z.require_tags
            ]
            for x, y in self.modules.items()
        }
        for cat in self.modules:
            it = self.modules[cat]
            order = [(m.name, m.after) for m in it]
            order = list(topologicalSort(order))
            self.modules[cat] = sorted(it, key=lambda x: order.index(x.name))

        for cat, it in self.modules.items():
            print(f"{str(cat)} -- {[x.name for x in it]}")
        if ModuleType.Output in self.modules and self.output_path is None:
            raise Exception("If using an output, must specify a path")

    def process(self, events):
        raw_event_count = ak.size(events, axis=0)
        dataset = events.metadata["dataset"]
        if ":" in dataset:
            dataset, set_name = dataset.split(":")
        else:
            set_name = dataset
        events["EventWeight"] = events.genWeight * self.weight_map[set_name]

        for module in self.modules.get(ModuleType.BaseObjectDef, []):
            events = module.func(events)

        for module in self.modules.get(ModuleType.PreSelectionProducer, []):
            events = module.func(events)

        for module in self.modules.get(ModuleType.PreSelectionHist, []):
            module.func(events, makeHistogram)

        if self.modules.get(ModuleType.Selection, False):
            selection = processor.PackedSelection()
            for module in self.modules.get(ModuleType.Selection, []):
                selection = module.func(events, selection)
                events = events[selection.all(*selection.names)]

        to_accumulate = []

        cat_data = {"CatDataset": dataset}

        x = zip(
            *(x.func(events, cat_data) for x in self.modules[ModuleType.Categories])
        )
        hm = makeCategoryHist(*x, events.EventWeight)

        for module in self.modules.get(ModuleType.MainProducer, []):
            events = module.func(events)

        for module in self.modules.get(ModuleType.MainHist, []):
            to_accumulate.append(module.func(events, hm))

        produced = []
        for module in self.modules.get(ModuleType.Output, []):
            produced.append(module.func(events, self.output_path))

        ret = {"dataset_info": {}}
        if to_accumulate:
            ret["histograms"] = accumulate(to_accumulate)
        ret["dataset_info"][dataset] = {
            events.metadata["filename"]: {
                "num_post_selection_events": ak.size(events, axis=0),
                "num_raw_events": raw_event_count,
            }
        }
        if produced:
            ret["dataset_info"][dataset]["produced"] = produced
        return ret

    def postprocess(self, accumulator):
        pass
