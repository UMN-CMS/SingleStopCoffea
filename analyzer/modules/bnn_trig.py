import awkward as ak
from pathlib import Path
import dask_awkward as da
from analyzer.core import MODULE_REPO, ModuleType
import json
import numpy as np
from coffea.ml_tools import numpy_call_wrapper


# class BNNEnsemble(numpy_call_wrapper):
#     def __init__(self, num_inputs, mean, sigma, hidden_width, data):
#         self.mean = mean
#         self.sigma = sigma
#         hidden_length = hidden_width * (num_inputs + 1)
#         hidden_data = data[:, 0:hidden_length].reshape(len(data), -1, num_inputs + 1)
#         self.output_bias = data[:, hidden_length]
#         self.output_weights = data[:, hidden_length + 1 :]
#         self.hidden_biases = hidden_data[:, :, 0]
#         self.hidden_weights = hidden_data[:, :, 1:]
#
#     def numpy_call(self, data):
#         data = (data - self.mean) / self.sigma
#         l1 = self.hidden_biases + np.matvec(self.hidden_weights, data[:, np.newaxis])
#         l2 = self.output_bias + np.vecdot(self.output_weights, np.tanh(l1))
#         vals = 1 / (1 + np.exp(-l2))
#         return np.median(vals,axis=1)
#         return np.stack(
#             [
#                 np.median(vals, axis=1),
#                 np.percentile(vals, 15.865, axis=1),
#                 np.percentile(vals, 84.135, axis=1),
#             ]
#         )
#
#     def prepare_awkward(self, data):
#         return [data], {}
#
#     @staticmethod
#     def fromFile(path):
#         with open(path, "r") as f:
#             data = json.load(f)
#         return BNNEnsemble(
#             data["num_inputs"],
#             np.array(data["mean"]),
#             np.array(data["sigma"]),
#             data["hidden_width"],
#             np.array(data["parameters"]),
#         )


class BNNEnsemble:
    def __init__(self, num_inputs, mean, sigma, hidden_width, data):
        self.mean = mean
        self.sigma = sigma
        hidden_length = hidden_width * (num_inputs + 1)
        hidden_data = data[:, 0:hidden_length].reshape(len(data), -1, num_inputs + 1)
        self.output_bias = data[:, hidden_length]
        self.output_weights = data[:, hidden_length + 1 :]
        self.hidden_biases = hidden_data[:, :, 0]
        self.hidden_weights = hidden_data[:, :, 1:]

    def __call__(self, data):
        data = data.to_numpy()
        data = (data - self.mean) / self.sigma
        l1 = self.hidden_biases + np.matvec(self.hidden_weights, data[:, None])
        l2 = self.output_bias + np.vecdot(self.output_weights, np.tanh(l1))
        vals = 1 / (1 + np.exp(-l2))
        return (
            np.median(vals, axis=1),
            np.percentile(vals, 15.865, axis=1),
            np.percentile(vals, 84.135, axis=1),
        )

    @staticmethod
    def fromFile(path):
        with open(path, "r") as f:
            data = json.load(f)
        return BNNEnsemble(
            data["num_inputs"],
            np.array(data["mean"]),
            np.array(data["sigma"]),
            data["hidden_width"],
            np.array(data["parameters"]),
        )


@MODULE_REPO.register(ModuleType.Weight)
def trigger_bnn(events, params, weight_manager, base_path=""):
    base_path = Path(base_path)
    era = params.dataset.era.name
    path = base_path / f"{era}.json"
    bnn = BNNEnsemble.fromFile(path)

    ht = events.HT
    fj = events.FatJet
    fjpt = fj[:, 0].pt
    fjmsd = fj[:, 0].msoftdrop
    out = bnn(ak.concatenate([x[:, np.newaxis] for x in [ht, fjpt, fjmsd]], axis=1))
    central, down, up = out[ 0], out[ 1], out[ 2]
    weight_manager.add(f"trigger_weight", central, {"bnn_unc": (up, down)})
    return
