import itertools as it

import awkward as ak
import numpy as np

from analyzer.core import analyzerModule
from analyzer.math_funcs import angleToNPiToPi

from .axes import *
from .utils import numMatching

from coffea.ml_tools.torch_wrapper import torch_wrapper
from analyzer.matching import object_matching

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def makeIdxHist(analyzer, idxs, name, axlabel):
    analyzer.H(
        name,
        hist.axis.IntCategory([0, 1, 2, 3, 4, 5, 6], name=name, label=axlabel),
        idxs,
        name=name,
    )

        
@analyzerModule(
    "reco_efficiency", categories="main", depends_on=["chargino_hists", "delta_r"]
)
def recoEfficiency(events, analyzer):
    ret = {}
    gj = events.good_jets
    all_three_mask = ~ak.any(ak.is_none(events.matched_jet_idx[:, 1:4], axis=1), axis=1)

    analyzer.H(
        f"m13_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M13|",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 0:3]),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m13_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M13| All Three Matched",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 0:3])[
            all_three_mask
        ],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    analyzer.H(
        f"m24_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M24|",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 1:4]),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m24_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ M24| All Three Matched",
        ),
        numMatching(events.matched_jet_idx[:, 1:4], ak.local_index(gj, axis=1)[:, 1:4])[
            all_three_mask
        ],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    analyzer.H(
        f"m3_top_3_no_lead_b_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top3NoB|",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b
        ),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m3_top_3_no_lead_b_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top3NoB| All Three Matched",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b
        )[all_three_mask],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )

    analyzer.H(
        f"m3_top_2_plus_lead_b_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top2PlusB|",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_2_plus_lead_b
        ),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m3_top_2_plus_lead_b_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ Top2PlusB| All Three Matched",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.top_2_plus_lead_b
        )[all_three_mask],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    analyzer.H(
        f"m3_dr_switched_matching",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ dr switched|",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.delta_r_switched
        ),
        name="Number of jets in this set that are also in the gen level matching",
    )
    analyzer.H(
        f"m3_dr_switched_matching_all_three",
        hist.axis.IntCategory(
            [0, 1, 2, 3],
            name="num_matched_chi",
            label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ dr switched| All Three Matched",
        ),
        numMatching(
            events.matched_jet_idx[:, 1:4], events.matching_algos.delta_r_switched
        )[all_three_mask],
        name="Number of jets in this set that are also in the gen level matching",
        mask=all_three_mask,
    )
    # analyzer.H(
    #     f"m3_top_3_no_lead_b_dr_cut_matching",
    #     hist.axis.IntCategory(
    #         [0, 1, 2, 3],
    #         name="num_matched_chi",
    #         label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ top 3 no lead b dr cut|",
    #     ),
    #     numMatching(
    #         events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b_dr_cut
    #     ),
    #     name="Number of jets in this set that are also in the gen level matching",
    # )
    # analyzer.H(
    #     f"m3_top_3_no_lead_b_dr_cut_matching_all_three",
    #     hist.axis.IntCategory(
    #         [0, 1, 2, 3],
    #         name="num_matched_chi",
    #         label=r"|GenMatcher($\Delta R < 0.2$) $\cap$ top 3 no lead b dr cut| All Three Matched",
    #     ),
    #     numMatching(
    #         events.matched_jet_idx[:, 1:4], events.matching_algos.top_3_no_lead_b_dr_cut
    #     )[all_three_mask],
    #     name="Number of jets in this set that are also in the gen level matching",
    #     mask=all_three_mask,
    # )
    return events, analyzer


@analyzerModule("combo_mass", depends_on=["chargino_hists"], categories="main")
def combo_method(events, analyzer):
    ret = {}
    gj = events.good_jets

    idx = ak.local_index(gj, axis=1)
    bwps = analyzer.profile.btag_working_points
    med_bjet_mask = gj.btagDeepFlavB > bwps["medium"]

    lead_b_idx = idx[med_bjet_mask][:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]

    jets_idx = ak.argcombinations(gj, 3, axis=1, replacement=False)
    import numpy as np

    one, two, three = ak.unzip(jets_idx)
    jets_idx = ak.concatenate(
        [one[:, :, np.newaxis], two[:, :, np.newaxis], three[:, :, np.newaxis]], axis=2
    )

    jets = ak.combinations(gj, 3, axis=1, replacement=False)
    one, two, three = ak.unzip(jets)
    jets = ak.concatenate(
        [one[:, :, np.newaxis], two[:, :, np.newaxis], three[:, :, np.newaxis]], axis=2
    )

    cross_jets = ak.argcartesian([jets, jets], axis=2)
    one, two = ak.unzip(cross_jets)
    max_delta_rs = ak.max(jets[one].delta_r(jets[two]), axis=2)

    dr_mask = max_delta_rs < 2
    lead_b_mask = (
        (jets_idx[:, :, 0] == lead_b_idx)
        | (jets_idx[:, :, 1] == lead_b_idx)
        | (jets_idx[:, :, 2] == lead_b_idx)
    )
    sublead_b_mask = (
        (jets_idx[:, :, 0] == sublead_b_idx)
        | (jets_idx[:, :, 1] == sublead_b_idx)
        | (jets_idx[:, :, 2] == sublead_b_idx)
    )

    summed = jets[:, :, 0] + jets[:, :, 1] + jets[:, :, 2]
    all_masses = summed.mass
    complete_mask = (~lead_b_mask) & dr_mask & sublead_b_mask
    just_b_mask = (~lead_b_mask) & sublead_b_mask
    uncomp_mass = ak.where(
        ak.any(complete_mask, axis=1),
        ak.firsts(all_masses[complete_mask]),
        all_masses[just_b_mask][:, 0],
    )
    uncomp_idx = ak.where(
        ak.any(complete_mask, axis=1),
        ak.firsts(jets_idx[complete_mask]),
        jets_idx[just_b_mask][:, 0],
    )

    analyzer.H(
        f"m3_top_3_no_lead_b_delta_r_cut",
        makeAxis(
            60,
            0,
            3000,
            r"mass of jets 1-3 without leading b, $\Delta R < 3$",
            unit="GeV",
        ),
        uncomp_mass,
        name="mass of jets 1-3 without leading b dr cut",
    )

    makeIdxHist(
        analyzer,
        uncomp_idx,
        "m3_top_3_no_lead_b_delta_r_cut_idx",
        "m3_top_3_no_lead_b_delta_r_cut_idxs",
    )
    events["matching_algos", "top_3_no_lead_b_dr_cut"] = uncomp_idx

    # combos = ak.argcombinations(list(range(6)), 3, axis=0)
    # padmass = ak.pad_none(all_masses, len(combos), axis=1)
    return events, analyzer


@analyzerModule("cat_combo_mass", depends_on=["chargino_hists"], categories="main")
def cat_combo_methods(events, analyzer):
    ret = {}
    gj = events.good_jets

    idx = ak.local_index(gj, axis=1)
    med_bjet_mask = gj.btagDeepFlavB > b_tag_wps[1]

    lead_b_idx = idx[med_bjet_mask][:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]
    lead_b_mask = ak.local_index(gj, axis=1) == lead_b_idx
    sublead_b_mask = ak.local_index(gj, axis=1) == sublead_b_idx

    m14 = gj[:, 0:4].sum().mass

    # m3_top_3_no_lead_b unless dRMax(3 others without lead b) / dRMax(3 others without sublead b) > 2
    # (to be optimized), in which case, put the leading b in m3.
    tol1 = 1

    gj_no_lead_b = gj[~lead_b_mask]
    gj_no_sublead_b = gj[~sublead_b_mask]

    cross_jets_no_lead_b = ak.cartesian([gj_no_lead_b[:, 0:3], gj_no_lead_b[:, 0:3]])
    delta_rs_no_lead_b = cross_jets_no_lead_b["0"].delta_r(cross_jets_no_lead_b["1"])
    delta_rs_no_lead_b = delta_rs_no_lead_b[delta_rs_no_lead_b > 0]
    max_delta_rs_no_lead_b = ak.max(delta_rs_no_lead_b, axis=1)

    cross_jets_no_sublead_b = ak.cartesian([gj_no_sublead_b[:, 0:3], gj_no_sublead_b[:, 0:3]])
    delta_rs_no_sublead_b = cross_jets_no_sublead_b["0"].delta_r(cross_jets_no_sublead_b["1"])
    delta_rs_no_sublead_b = delta_rs_no_sublead_b[delta_rs_no_sublead_b > 0]
    max_delta_rs_no_sublead_b = ak.max(delta_rs_no_sublead_b, axis=1)

    # print((max_delta_rs_no_lead_b / max_delta_rs_no_sublead_b))

    no_lead_b_in_m3_mask = ((max_delta_rs_no_lead_b / max_delta_rs_no_sublead_b) > tol1)
    uncomp_charg_jets_1 = ak.where(no_lead_b_in_m3_mask, gj_no_sublead_b[:, 0:3], gj_no_lead_b[:, 0:3])

    uncomp_charg_mass_1 = uncomp_charg_jets_1.sum().mass

    analyzer.H(
        f"max_delta_rs_no_lead_b_over_no_sublead_b",
        makeAxis(
            50,
            0,
            10,
            rf"max_delta_rs_no_lead_b divided by max_delta_rsno_sublead_b",
            unit="GeV",
        ),
        max_delta_rs_no_lead_b / max_delta_rs_no_sublead_b,
        name="max_delta_rs_no_lead_b divided by max_delta_rsno_sublead_b",
    )

    analyzer.H(
        f"m3_top_3_no_lead_b_delta_r_cut",
        makeAxis(
            60,
            0,
            3000,
            rf"mass of jets 1-3 without leading b if $\Delta R$ ratio < {tol1}",
            unit="GeV",
        ),
        uncomp_charg_mass_1,
        name="mass of jets 1-3 without leading b dr cut",
    )

    analyzer.H(
        f"m14_vs_m3_top_3_no_lead_b_delta_r_cut",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, rf"mass of jets 1-3 without leading b if $\Delta R$ ratio < {tol1}", unit="GeV"),
        ],
        [m14, uncomp_charg_mass_1],
        name="$m_{14}$ vs Mass of Jets 1-3 Without Leading B",
    )


    # Reserve the subleading 3 jets for m3 as we do now unless dRMax(m3) > tol, in which case, see if swapping
    # j5 for j4 gives dRMan(m3) < tol. If not, see if j6 works. If none do, default back to j4.
    tol2 = 2
    gj234 = gj[:, 1:4]
    padded_gj = ak.pad_none(gj, 6)
    mask_235 = ak.is_none(padded_gj, axis=1)[:, 4]
    mask_235 = mask_235[:, np.newaxis]
    mask_236 = ak.is_none(padded_gj, axis=1)[:, 5]
    mask_236 = mask_236[:, np.newaxis]
    gj235 = ak.where(mask_235, padded_gj[:, [1,2,3]], padded_gj[:, [1,2,4]])
    gj236 = ak.where(mask_236, padded_gj[:, [1,2,3]], padded_gj[:, [1,2,5]])

    cross_jets_234 = ak.cartesian([gj234[:, 0:3], gj234[:, 0:3]])
    cross_jets_235 = ak.cartesian([gj235[:, 0:3], gj235[:, 0:3]])
    cross_jets_236 = ak.cartesian([gj236[:, 0:3], gj236[:, 0:3]])
    delta_rs_234 = cross_jets_234["0"].delta_r(cross_jets_234["1"])
    delta_rs_235 = cross_jets_235["0"].delta_r(cross_jets_235["1"])
    delta_rs_236 = cross_jets_236["0"].delta_r(cross_jets_236["1"])
    max_delta_rs_234 = ak.max(delta_rs_234, axis=1)
    max_delta_rs_235 = ak.max(delta_rs_235, axis=1)
    max_delta_rs_236 = ak.max(delta_rs_236, axis=1)
    
    drmask_234 = (max_delta_rs_234 < tol2)
    drmask_234 = drmask_234[:, np.newaxis]
    drmask_235 = (max_delta_rs_235 < tol2)
    drmask_235 = drmask_235[:, np.newaxis]
    drmask_236 = (max_delta_rs_236 < tol2)
    drmask_236 = drmask_236[:, np.newaxis]

    uncomp_charg_jets_2 = ak.where(drmask_234, gj234, ak.where(drmask_235, gj235, ak.where(drmask_236, gj236, gj234)))
    uncomp_charg_mass_2 = uncomp_charg_jets_2.sum().mass

    analyzer.H(
        f"m3_234_235_236_delta_r_cut",
        makeAxis(
            60,
            0,
            3000,
            rf"the j4 j5 j6 one. $\Delta R < {tol1}$",
            unit="GeV",
        ),
        uncomp_charg_mass_2,
        name="x",
    )

    analyzer.H(
        f"m14_vs_m3_234_235_236_delta_r_cut",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, rf"the j4 j5 j6 one. $\Delta R < {tol2}$", unit="GeV"),
        ],
        [m14, uncomp_charg_mass_2],
        name=rf"m14_vs_m3_234_235_236_delta_r_cut$",
    )
    
    # Reserve the leading b for m3. Then the other two are the leading 2 of the remaining jets.
    comp_charg_mass_1 = ak.flatten(gj_no_lead_b[:, :2].sum() + gj[lead_b_mask]).mass
    analyzer.H(
        f"m3_lead_b_and_other_two_lead",
        makeAxis(
            60,
            0,
            3000,
            rf"m3_lead_b_and_other_two_lead",
            unit="GeV",
        ),
        comp_charg_mass_1,
        name="m3_lead_b_and_other_two_lead",
    )
    analyzer.H(
        f"m14_vs_m3_lead_b_and_other_two_lead",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, rf"m3_lead_b_and_other_two_lead", unit="GeV"),
        ],
        [m14, comp_charg_mass_1],
        name=rf"m14_vs_m3_lead_b_and_other_two_lead",
    )

    # Reserve the leading b among jets 4, 5, and 6 as the other m4 jet. If none are b’s, go to 3,
    # then 2. m3 is then the leading 3 remaining.
    padded_idx = ak.pad_none(idx, 6)
    padded_med_bjet_mask_reordered = ak.fill_none(ak.pad_none(med_bjet_mask, 6), False)[:, [3,4,5,2,1]]
    b_idx_reordered = padded_idx[:, [3,4,5,2,1]]

    inter = ak.mask(b_idx_reordered, padded_med_bjet_mask_reordered)
    inter_mask = ak.drop_none(inter)[:, 0]

    gj_sans_specific_b = gj[(idx != inter_mask)]
    comp_charg_mass_2 = gj_sans_specific_b[:, :3].sum().mass
    analyzer.H(
        f"m3_comp_alg_2",
        makeAxis(
            60,
            0,
            3000,
            rf"m3_comp_alg_2",
            unit="GeV",
        ),
        comp_charg_mass_2,
        name="\'Reserve the leading b among jets 4, 5, and 6...\'",
    )
    analyzer.H(
        f"m14_vs_m3_comp_alg_2",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, rf"m3_comp_alg_2", unit="GeV"),
        ],
        [m14, comp_charg_mass_2],
        name=rf"m14_vs_m3_comp_alg_2",
    )

    return events, analyzer


scl = open('scaler.pkl', 'rb')
scaler = pickle.load(scl)


class jetAssignmentNN(torch_wrapper):
    def prepare_awkward(self,events):
        # ak = self.get_awkward_lib(events)
        jets = events.good_jets
        flat_jets = ak.flatten(jets)

        m3 = jets[:,1:4].sum()
        m4 = jets[:,0:4].sum()

        ones = ak.ones_like(jets.pt)

        imap = {
            "features": {
                "jetOrdinality":    ak.flatten(ak.local_index(jets, axis=1)),
                "jetPT": 		    flat_jets.pt - 2,
                "jetEta": 		    flat_jets.eta,
                "jetPhi": 		    flat_jets.phi,
                "jetBScore":    	flat_jets.btagDeepFlavB,
                "m3M": 			    ak.flatten(ones * m3.mass),
                "m3PT": 		    ak.flatten(ones * m3.pt),
                "m3Eta": 		    ak.flatten(ones * m3.eta),
                "m3Phi": 	        ak.flatten(ones * m3.phi),
                "m4M": 			    ak.flatten(ones * m4.mass),
                "m4PT":		        ak.flatten(ones * m4.pt),
                "m4Eta":		    ak.flatten(ones * m4.eta),
                "m4Phi":	        ak.flatten(ones * m4.phi)
            }
        }
        
        imap_concat = ak.concatenate([x[:, np.newaxis] for x in imap['features'].values()], axis=1)
        imap_scaled = (imap_concat - scaler.mean_) / scaler.scale_
        return (ak.values_astype(imap_scaled, "float32"),),{}
    

b_tag_wps = [0.0490, 0.2783, 0.7100]

@analyzerModule("NN_mass", categories="main")
def NN_mass_reco(events, analyzer):
    jets = events.good_jets
    model_1500_900 = jetAssignmentNN("traced_model.pt")
    model_uncomp = jetAssignmentNN("jetMatcherNN_100_strat_newcuts_traced.pt")
    outputs_1500_900 = model_1500_900(events)
    outputs_uncomp = model_uncomp(events)

    m14 = jets[:, 0:4].sum().mass

    # just 1500_900
    high_charg_score_mask_1500_900 = ak.unflatten(outputs_1500_900[:,1] > 0.95, ak.num(jets))
    highest_3_charg_score_idx_1500_900 = ak.argsort(ak.unflatten(outputs_1500_900[:,1], ak.num(jets)), axis=1)[:, -3:]
    highest_stop_score_idx_1500_900 = ak.argsort(ak.unflatten(outputs_1500_900[:,0], ak.num(jets)), axis=1)[:, -1]

    top_3_charg_score_sum_1500_900 = jets[highest_3_charg_score_idx_1500_900].sum()
    m3_top_3_nn_charg_score_1500_900 = top_3_charg_score_sum_1500_900.mass
    m3_high_nn_charg_score_1500_900 = jets[high_charg_score_mask_1500_900].sum().mass

    stop_jets_1500_900 = jets[ak.singletons(highest_stop_score_idx_1500_900)]
    m4_nn = ak.flatten((top_3_charg_score_sum_1500_900 + stop_jets_1500_900).mass)

    # full uncompressed model
    high_charg_score_mask_uncomp = ak.unflatten(outputs_uncomp[:,1] > 0.95, ak.num(jets))
    highest_3_charg_score_idx_uncomp = ak.argsort(ak.unflatten(outputs_uncomp[:,1], ak.num(jets)), axis=1)[:, -3:]
    highest_stop_score_idx_uncomp = ak.argsort(ak.unflatten(outputs_uncomp[:,0], ak.num(jets)), axis=1)[:, -1]

    top_3_charg_score_sum_uncomp = jets[highest_3_charg_score_idx_uncomp].sum()
    m3_top_3_nn_charg_score_uncomp = top_3_charg_score_sum_uncomp.mass
    m3_high_nn_charg_score_uncomp = jets[high_charg_score_mask_uncomp].sum().mass

    stop_jets_uncomp = jets[ak.singletons(highest_stop_score_idx_uncomp)]
    m4_nn = ak.flatten((top_3_charg_score_sum_uncomp + stop_jets_uncomp).mass)
    
    # hists
    analyzer.H(
        f"m3_top_3_nn_charg_score_1500900",
        makeAxis(
            60,
            0,
            3000,
            rf"m3_top_3_nn_charg_score_1500_900",
            unit="GeV",
        ),
        m3_top_3_nn_charg_score_1500_900,
        name="\'Mass of sum of highest-scoring jets according to chargino jet small NN classifier\'",
    )

    analyzer.H(
        f"m3_top_3_nn_charg_score_uncomp",
        makeAxis(
            60,
            0,
            3000,
            rf"m3_top_3_nn_charg_score_uncomp",
            unit="GeV",
        ),
        m3_top_3_nn_charg_score_uncomp,
        name="\'Mass of sum of highest-scoring jets according to chargino jet big NN classifier\'",
    )
    
    analyzer.H(
        f"m3_high_nn_charg_score",
        makeAxis(
            60,
            0,
            3000,
            rf"m3_high_nn_charg_score",
            unit="GeV",
        ),
        m3_high_nn_charg_score_uncomp,
        name="\'Mass of sum of all jets with chargino score above 0.8\'",
    )
    analyzer.H(
        f"m14_vs_m3_top_3_nn_charg_score",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, r"$m_{3 (NN)}$", unit="GeV"),
        ],
        [m14, m3_top_3_nn_charg_score_uncomp],
        name="$m_{14}$ vs Mass of sum of highest-scoring jets according to chargino jet NN classifier",
    )
    analyzer.H(
        f"m4nn_vs_m3_top_3_nn_charg_score",
        [
            makeAxis(60, 0, 3000, r"$m_{4 (NN)}$", unit="GeV"),
            makeAxis(60, 0, 3000, r"$m_{3 (NN)}$", unit="GeV"),
        ],
        [m4_nn, m3_top_3_nn_charg_score_uncomp],
        name="$m_{4 (NN)}$ vs Mass of sum of highest-scoring jets according to chargino jet NN classifier",
    )
    return events, analyzer

@analyzerModule("chargino_hists", categories="main")
def charginoRecoHistograms(events, analyzer):
    ret = {}
    gj = events.good_jets

    idx = ak.local_index(gj, axis=1)

    bwps = analyzer.profile.btag_working_points
    med_bjet_mask = gj.btagDeepFlavB > bwps["medium"]

    t_lead_b_idx = idx[med_bjet_mask]
    lead_b_idx = t_lead_b_idx[:, 0]
    sublead_b_idx = idx[med_bjet_mask][:, 1]
    no_lead_idxs = idx[idx != lead_b_idx]
    no_sublead_idxs = idx[idx != sublead_b_idx]
    no_lead_or_sublead_idxs = idx[(idx != sublead_b_idx) & (idx != lead_b_idx)]
    no_lead_jets = gj[no_lead_idxs]
    no_sublead_jets = gj[no_sublead_idxs]
    first, second = ak.unzip(ak.combinations(no_lead_jets[:, 0:3], 2))
    max_dr_no_lead = ak.max(first.delta_r(second), axis=1)
    first, second = ak.unzip(ak.combinations(no_sublead_jets[:, 0:3], 2))
    max_dr_no_sublead = ak.max(first.delta_r(second), axis=1)
    max_no_lead_over_max_sublead = max_dr_no_lead / max_dr_no_sublead

    jets = gj[:, 0:4].sum()
    m14 = jets.mass
    uncomp_charg_idxs = no_lead_idxs[:, 0:3]
    uncomp_charg = (no_lead_jets[:, 0:3].sum()).mass
    m14_axis = makeAxis(60, 0, 3000, r"$m_{14}$ [GeV]")
    mchi_axis = makeAxis(60, 0, 3000, r"$m_{\chi}$ [GeV]")

    analyzer.H(
        f"m3_top_3_no_lead_b",
        makeAxis(60, 0, 3000, r"mass of jets 1-3 without leading b", unit="GeV"),
        uncomp_charg,
        name="mass of jets 1-3 without leading b",
    )
    analyzer.H(
        f"m14_vs_m3_top_3_no_lead_b",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(60, 0, 3000, r"$m_{3 (no b)}$", unit="GeV"),
        ],
        [m14, uncomp_charg],
        name="$m_{14}$ vs Mass of Jets 1-3 Without Leading B",
    )
    analyzer.H(
        f"m3_top_3_no_lead_b_pt_diff",
        makeAxis(40, -400, 400, r"pt(charg)-pt(m4 w/o charg)", unit="GeV"),
        no_lead_jets[:, 0:3].sum().pt - ak.ravel(gj[ak.singletons(lead_b_idx)].pt),
        name="pt(charg)-pt(m4 w/o charg)",
    )
    # ret[f"m3_top_3_no_b_unless_dR_charg_gt_2"] = hmaker(
    #    mchi_axis,
    #    ak.where(
    #        max_no_lead_over_max_sublead > 2,
    #        (no_lead_jets[:, 0:3].sum()).mass,
    #        (
    #            gj[no_lead_or_sublead_idxs][:, 0:2].sum()
    #            + gj[ak.singletons(lead_b_idx)][:, 0]
    #        ).mass,
    #    ),
    #    name="m3_top3_no_b_unless_dR_charg_gt_2",
    # )

    comp_charg = (no_lead_jets[:, 0:2].sum() + gj[ak.singletons(lead_b_idx)][:, 0]).mass
    comp_charg_idxs = ak.concatenate(
        [no_lead_idxs[:, 0:2], ak.singletons(lead_b_idx)], axis=1
    )

    one, two = ak.unzip(ak.cartesian([no_lead_jets[:, 0:3], no_lead_jets[:, 0:3]]))
    dr = one.delta_r(two)
    max_dr_no_lead_b = ak.max(dr, axis=1)
    one, two = ak.unzip(
        ak.cartesian([gj[no_sublead_idxs[:, 0:3]], gj[no_sublead_idxs[:, 0:3]]], axis=1)
    )
    dr = one.delta_r(two)
    max_dr_no_sublead_b = ak.max(dr, axis=1)
    ratio = (max_dr_no_lead_b / max_dr_no_sublead_b) > 2

    decided = ak.where(ratio, comp_charg, uncomp_charg)
    decided_idxs = ak.where(ratio, comp_charg_idxs, uncomp_charg_idxs)

    events["matching_algos"] = ak.zip(
        dict(
            delta_r_switched=decided_idxs,
            top_2_plus_lead_b=comp_charg_idxs,
            top_3_no_lead_b=uncomp_charg_idxs,
        )
    )

    makeIdxHist(
        analyzer,
        uncomp_charg_idxs,
        "m3_top_3_no_lead_b_idxs",
        "m3_top_3_no_lead_b_idxs",
    )
    makeIdxHist(
        analyzer,
        comp_charg_idxs,
        "m3_top_2_plus_lead_b_idxs",
        "m3_top_2_plus_lead_b_idxs",
    )
    #print(comp_charg_idxs)
    #print(decided_idxs)
    #makeIdxHist(
    #    analyzer, decided_idxs, "m3_dr_switched_idxs", "m3_dr_switched_idxs"
    #)

    analyzer.H(
        f"m3_top_2_plus_lead_b",
        makeAxis(
            60, 0, 3000, r"Mass of leading 2 $p_{T}$ Jets + leading b Jet", unit="GeV"
        ),
        comp_charg,
        name="m3_top_2_plus_lead_b",
    )

    analyzer.H(
        f"m3_dr_switched",
        makeAxis(60, 0, 3000, r"$\Delta R$>2 Switched Mass", unit="GeV"),
        comp_charg,
        name="m3_top_2_plus_lead_b_delta_r_switch",
    )

    analyzer.H(
        f"m14_vs_m3_top_2_plus_lead_b",
        [
            makeAxis(60, 0, 3000, r"$m_{4}$", unit="GeV"),
            makeAxis(
                60,
                0,
                3000,
                r"Mass of leading 2 $p_{T}$ Jets + leading b Jet",
                unit="GeV",
            ),
        ],
        [m14, comp_charg],
        name="Mass of Top 2 $p_T$ Jets Plus Leading b Jet",
    )
    ratio_axis = hist.axis.Regular(
        50,
        0,
        1,
        name=f"ratio",
        label=r"$\frac{m_{\chi}}{m_{4}}$ [GeV]",
    )

    analyzer.H(
        f"ratio_m14_vs_m3_top_2_plus_lead_b",
        [
            makeAxis(
                60,
                0,
                3000,
                r"$m_{4}$",
                unit="GeV",
            ),
            makeAxis(
                50,
                0,
                1,
                r"$\frac{m_{3 (incl b)}}{m_{4}}$",
            ),
        ],
        [m14, comp_charg / m14],
        name="ratio_m14_vs_m3_top_2_plus_lead_b",
    )

    analyzer.H(
        f"ratio_m14_vs_m3_top_3_no_lead_b",
        [
            makeAxis(
                60,
                0,
                3000,
                r"$m_{4}$",
                unit="GeV",
            ),
            makeAxis(
                50,
                0,
                1,
                r"$\frac{m_{3 \mathrm{(no b)}}}{m_{4}}$",
            ),
        ],
        [m14, uncomp_charg / m14],
        name="ratio_m3_top_3_no_lead_b",
    )

    return events, analyzer


# class jetAssignmentNN(torch_wrapper):
#     def prepare_awkward(self,events):

#         awk = self.get_awkward_lib(events)
#         #jets = ak.flatten(events.good_jets)
#         jets = events.good_jets
#         m3 = jets[:,1:4].sum()
#         m4 = jets[:,0:4].sum()

#         imap = {
#             "features": {
#                 "jetOrdinality":	ak.local_index(jets),
#                 "jetPT": 		jets.pt,
#                 "jetEta": 		jets.eta,
#                 "jetPhi": 		jets.phi,
#                 "jetBScore": 		jets.btagDeepFlavB,
#                 "m3M": 			m3.mass,
#                 "m3PT": 		m3.pt,
#                 "m3Eta": 		m3.eta,
#                 "m3Phi": 		m3.phi,
#                 "m4M": 			m4.mass,
#                 "m4PT":			m4.pt,
#                 "m4Eta":		m4.eta,
#                 "m4Phi":		m4.phi,
#             }
#         }

#         return(),{
#             "features": awk.values_astype(imap["features"],"float32")
#         }
    
# @analyzerModule("jetAssignmentNN")
# def addNNScores(events, analyzer):
#     model = jetAssignmentNN("jetMatcherNN.pt")
#     scores = model(events)
#     print(scores)
#     events["NNStopProb"]  = scores[:,0]
#     events["NNChiProb"]   = scores[:,1]
#     events["NNOtherProb"] = scores[:,2] 
#     return events
#     analyzer.H(
#         f"m14_vs_m3NN",
#         [
#             makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
#             makeAxis(50, 0, 500, r"$pt_{14}$", unit="GeV"),
#         ],
#         [jets.mass, jets.pt],
#         name="m14 vs pt 14",
#     )


@analyzerModule("stop_reco", categories="main")
def stopreco(events, analyzer):
    ret = {}
    jets = events.good_jets[:, 0:4].sum()
    analyzer.H(
        f"m14_vs_pt14",
        [
            makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
            makeAxis(50, 0, 500, r"$pt_{14}$", unit="GeV"),
        ],
        [jets.mass, jets.pt],
        name="m14 vs pt 14",
    )

    padded_jets = ak.pad_none(events.good_jets, 6, axis=1)

    top5sum = padded_jets[:, 0:5].sum()
    top6sum = padded_jets[:, 0:6].sum()

    fsrincluded = ak.where((top5sum.pt < jets.pt), top5sum.mass, jets.mass)

    analyzer.H(
        "m14_gt100_m15",
        makeAxis(60, 0, 3000, r"$m_{14}$", unit="GeV"),
        fsrincluded,
        name="m14 or maybe m15",
    )

    return events, analyzer
