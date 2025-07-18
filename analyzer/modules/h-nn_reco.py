import pickle
import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType
from coffea.ml_tools.torch_wrapper import torch_wrapper
import numpy as np

@MODULE_REPO.register(ModuleType.Producer)
def H_NN_reco(columns, params, model_path=None, scaler_path=None):
    class ABCDiHiggsNetwork(torch_wrapper):
        def prepare_awkward(self, columns, scalerFile):
            def pad(arr):
                return ak.fill_none(
                    ak.pad_none(arr, 4, axis=1, clip=True),
                    0.0,
                )
            with open(scalerFile, "rb") as f:
                scaler = pickle.load(f)
            jets = columns.Jet
            electrons = columns.Electron
            muons = columns.Muon
            HT = ak.sum(jets.pt, axis=1)
            imap = {
                "features": {
                    "HT": HT,
                    "elept": pad(ak.topk(electrons, 4, key="pt").pt),
                    "mupt": pad(ak.topk(muons, 4, key="pt").pt),
                },
            }
            imap_concat = ak.concatenate([x[:, np.newaxis, :] for x in imap["features"].values()], axis=1)
            imap_scaled = (imap_concat - scaler.mean_) / scaler.scale_
            return (), {
                "features": ak.values_astype(imap_scaled, "float32"),
            }
    model = ABCDiHiggsNetwork(model_path)
    outputs = model(columns, scaler_path)
    columns.add("Disc1", outputs[0])
    columns.add("Disc2", outputs[1])
    

    