import pickle
import awkward as ak
from analyzer.core import MODULE_REPO, ModuleType
from coffea.ml_tools.torch_wrapper import torch_wrapper
import numpy as np
import torch
from .utils.btag_points import getBTagWP
from .utils.axes import makeAxis


@MODULE_REPO.register(ModuleType.Histogram)
def h_nn(events, params, analyzer, model_path=None, scaler_path=None):
    # Define inputs
    bwps = getBTagWP(params)
    good_jets = events.good_jets
    medb_mask = good_jets.btagDeepFlavB > bwps["M"]
    med_bs = good_jets[medb_mask]

    local_indices = ak.local_index(good_jets, axis=1)
    toptwob_indices = local_indices[medb_mask][:,:2]
    ind_1 = toptwob_indices[:,0]
    ind_2 = toptwob_indices[:,1]
    non2b_mask = (local_indices != ind_1) & (local_indices != ind_2)
    non2b = good_jets[non2b_mask]
    events.add("non2b_HT", ak.sum(non2b.pt, axis=1))
    events.add("non2b_fourjet", non2b[:,:4].sum())

    bjet1 = med_bs[:,0]
    bjet2 = med_bs[:,1]
    bsum = bjet1 + bjet2

    events.add("b_dijet", bjet1 + bjet2)
    events.add("b_deltaR", bjet1.delta_r(bjet2))   
    events.add("b_HT", ak.sum(med_bs.pt, axis=1))
    
    non2b_dijet_combos = ak.combinations(non2b[:,0:4], 2, fields=["jet1", "jet2"])
    non2b_dijets_mass = (non2b_dijet_combos.jet1 + non2b_dijet_combos.jet2).mass
    w_diff = abs(non2b_dijets_mass - 80.3692)
    sorted_indices = ak.argsort(w_diff, axis=1, ascending=True)
    w_diff_subleast = w_diff[sorted_indices][:,1]
    events.add("w_diff_subleast", w_diff_subleast)

    b1_drs = bjet1.delta_r(non2b)
    b1_sorted_idx = ak.argsort(b1_drs, axis=1)
    b1_two_nearest = non2b[b1_sorted_idx]
    b1_trijet = bjet1 + b1_two_nearest[:,0] + b1_two_nearest[:,1]
    events.add("b1_trijet", b1_trijet)
    
    b2_drs = bjet2.delta_r(non2b)
    b2_sorted_idx = ak.argsort(b2_drs[:,:2], axis=1)
    b2_two_nearest = non2b[b2_sorted_idx[:, :2]]
    b2_trijet = bjet2 + b2_two_nearest[:,0] + b2_two_nearest[:,1]
    events.add("b2_trijet", b2_trijet)

    # Evaluate Model
    class ABCDiHiggsNetwork(torch_wrapper):
        def prepare_awkward(self, inputs):
            return [ak.values_astype(inputs, "float32"),], {}

        def postprocess_awkward(self, output, events):
            return {
                "Disc1": output[:, 0],
                "Disc2": output[:, 1],
            }


    if model_path is None or scaler_path is None:
        raise ValueError("NN Requires a model path and scaler path")

    jet_variables = [
        "good_jets_pt",
        "good_jets_eta",
        "good_jets_phi",
        "good_jets_mass",
        "good_jets_btagDeepFlavB",
    ]
    global_variables = [
        "HT",
        "b_HT",
        "b_deltaR",
        "non2b_HT",
        "w_diff_subleast",
    ]
    fourvecvars = ["pt", "eta", "phi", "mass"]
    global_variables += [f"b_dijet_{var}" for var in fourvecvars]
    global_variables += [f"b1_trijet_{var}" for var in fourvecvars]
    global_variables += [f"b2_trijet_{var}" for var in fourvecvars]
    global_variables += [f"non2b_fourjet_{var}" for var in fourvecvars]
    model = ABCDiHiggsNetwork(model_path)

    def makeNNVariableAxis(var):
        vecvar, topvar = None, None
        fourvecvars = ["pt", "eta", "phi", "mass", "btagDeepFlavB"]
        if any([fourvecvar in var for fourvecvar in fourvecvars]):
            vecvar, topvar = var.split("_")[-1], "_".join(var.split("_")[:-1])
        retdict = {
            "btagDeepFlavB": (25, 0, 1, rf"{topvar} btagDeepFlavB Score", None),
            "pt": (25, 0, 500, rf"{topvar} $p_T$", "GeV"),
            "eta": (25, -2.5, 2.5, rf"{topvar} $\eta$", None),
            "phi": (25, -4, 4, rf"{topvar} $\phi$", None),
            "mass": (25, 0, 400, rf"{topvar} Mass", "GeV"),
            "HT": (25, 350, 1250, r"Jet $H_T$", "GeV"),
            "b_HT": (25, 0, 500, r"b-Jet $H_T$", "GeV"),
            "b_deltaR": (25, 0, 5, r"b-Jet $\Delta$R", None),
            "non2b_HT": (25, 0, 800, r"non2b-Jet $H_T$", "GeV"),
            "w_diff_subleast": (25, 0, 75, r"Subleading $\Delta M_{W}$", "GeV")
        }
        bins, start, stop, label, unit = retdict[var] if vecvar is None else retdict[vecvar]
        return makeAxis(bins, start, stop, label, unit)

    jet_plotting = {}
    jet_features = []
    for var in jet_variables:
        vecvar, topvar = var.split("_")[-1], "_".join(var.split("_")[:-1])
        for jetidx in range(6):
            jet_var = events[topvar][vecvar][:, jetidx]
            jet_features.append(jet_var[:, np.newaxis])
            jet_plotting[f"{topvar}_{jetidx}_{vecvar}"] = jet_var
    global_plotting = {}
    global_features = []
    for var in global_variables:
        if any([fourvecvar in var for fourvecvar in fourvecvars]):
            vecvar, topvar = var.split("_")[-1], "_".join(var.split("_")[:-1])
            global_var = getattr(events[topvar], vecvar)
        else:
            vecvar, topvar = None, var
            global_var = events[topvar]
        global_plotting[var] = global_var
        global_features.append(global_var[:, np.newaxis])
        
    X_plotting = jet_plotting | global_plotting
    X_features = jet_features + global_features
    X = ak.concatenate(X_features, axis=1)
    with open(scaler_path, "rb") as f:
        scaler = torch.load(f, map_location="cpu")["scaler"]
    X = (X - scaler.mean_) / scaler.scale_
    outputs = model(X)

    disc1, disc2 = outputs["Disc1"], outputs["Disc2"]
    analyzer.H(
        "disc1",
        makeAxis(50, 0, 1, "Disc. 1 Score"),
        disc1,
        description="NN discriminator 1 score"
    )
    analyzer.H(
        "disc2",
        makeAxis(50, 0, 1, "Disc. 2 Score"),
        disc2,
        description="NN discriminator 2 score"
    )
    analyzer.H(
        "abcdplane",
        [makeAxis(50, 0, 1, "Disc. 1 Score"), makeAxis(50, 0, 1, "Disc. 2 Score")],
        [disc1, disc2],
        description="ABCD plane 2d plot"
    )
    for var, feature in X_plotting.items():
        analyzer.H(
            var,
            makeNNVariableAxis(var),
            feature,
            description=f"NN input {var}"
        )
        analyzer.H(
            f"d1_v_{var}",
            [makeNNVariableAxis(var), makeAxis(50, 0, 1, "Disc. 1 Score")],
            [feature, disc2],
            description=f"NN input {var} against disc1"
        )
        analyzer.H(
            f"d2_v_{var}",
            [makeNNVariableAxis(var), makeAxis(50, 0, 1, "Disc. 2 Score")],
            [feature, disc2],
            description=f"NN input {var} against disc2"
        )



        