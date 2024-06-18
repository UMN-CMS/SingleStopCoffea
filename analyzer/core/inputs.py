import logging
import uproot
import analyzer.utils as utils
from coffea.dataset_tools.preprocess import DatasetSpec
import coffea.dataset_tools as dst
from coffea.dataset_tools.preprocess import DatasetSpec
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Optional,
    Set,
    Tuple,
    Union,
)


logger = logging.getLogger(__name__)

@dataclass
class AnalyzerInput:
    dataset_name: str
    fill_name: str
    coffea_dataset: DatasetSpec
    profile: Any
    lumi_json: Optional[str] = None

@dataclass
class DatasetPreprocessed:
    dataset_input: AnalyzerInput
    coffea_dataset_split: DatasetSpec

    @staticmethod
    def fromDatasetInput(dataset_input, **kwargs):
        out, x = dst.preprocess(dataset_input.coffea_dataset, save_form=False, uproot_options=dict(timeout=120), skip_bad_files=True, **kwargs)
        return DatasetPreprocessed(dataset_input, out[dataset_input.dataset_name])

    def getCoffeaDataset(self) -> DatasetSpec:
        return self.coffea_dataset_split


def preprocessBulk(dataset_input: Iterable[AnalyzerInput], **kwargs):
    mapping = {x.dataset_name: x for x in dataset_input}
    all_inputs = utils.accumulate([x.coffea_dataset for x in dataset_input])
    out, x = dst.preprocess(all_inputs, uproot_options=dict(timeout=120), skip_bad_files=True, **kwargs)
    ret = [DatasetPreprocessed(mapping[k], v) for k, v in out.items()]
    return ret
