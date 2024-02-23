#from coffea.ml_tools.torch_wrapper import torch_wrapper
from .torch_wrapper import torch_wrapper
from analyzer.core import analyzerModule, ModuleType
from analyzer.matching import object_matching
import awkward as ak

class jetAssignmentNN(torch_wrapper):
    def prepare_awkward(self,events):

        awk = self.get_awkward_lib(events)
        #jets = ak.flatten(events.good_jets)
        jets = events.good_jets
        m3 = jets[:,1:4].sum()
        m4 = jets[:,0:4].sum()

        imap = {
            "features": {
                "jetOrdinality":	ak.local_index(jets),
                "jetPT": 		jets.pt,
                "jetEta": 		jets.eta,
                "jetPhi": 		jets.phi,
                "jetBScore": 		jets.btagDeepFlavB,
                "m3M": 			m3.mass,
                "m3PT": 		m3.pt,
                "m3Eta": 		m3.eta,
                "m3Phi": 		m3.phi,
                "m4M": 			m4.mass,
                "m4PT":			m4.pt,
                "m4Eta":		m4.eta,
                "m4Phi":		m4.phi,
            }
        }

        return(),{
            "features": awk.values_astype(imap["features"],"float32")
        }

@analyzerModule("event_level", ModuleType.MainProducer)
def addEventLevelVars(events):
    ht = ak.sum(events.good_jets.pt, axis=1)
    events["HT"] = ht
    return events


@analyzerModule("delta_r", ModuleType.MainProducer,require_tags=["signal"], after=["good_gen"])
def deltaRMatch(events):
    # ret =  object_matching(events.SignalQuarks, events.good_jets, 0.3, None, False)
    matched_jets, matched_quarks, dr, idx_j, idx_q, _ = object_matching(
        events.good_jets, events.SignalQuarks, 0.3, 0.5, True
    )
    events["matched_quarks"] = matched_quarks
    events["matched_jets"] = matched_jets
    events["matched_dr"] = dr
    events["matched_jet_idx"] = idx_j
    return events

@analyzerModule("jetAssignmentNN", ModuleType.MainProducer)
def addNNScores(events):
    model = jetAssignmentNN("/uscms_data/d3/dmahon/NanoAODTools/CMSSW_10_6_19_patch2/src/PhysicsTools/NanoAODTools/python/postprocessing/singleStop/output/jetMatcherNNPyTorch/jetMatcherNN.pt")
    scores = model(events)
    print(scores)
    events["NNStopProb"]  = scores[:,0]
    events["NNChiProb"]   = scores[:,1]
    events["NNOtherProb"] = scores[:,2] 
    return events
