import itertools as it

import awkward as ak
import hist
from analyzer.core import MODULE_REPO, ModuleType

from .utils.axes import CommonAxes, makeAxis


@MODULE_REPO.register(ModuleType.Histogram)
def njets(events, params, analyzer):
    """Basic information about individual jets"""
    gj = events.good_jets
    analyzer.H(f"h_njet", CommonAxes.nj_axis, ak.num(gj))
    #jet_combos = [(0, 4), (0, 3), (1, 4)]


@MODULE_REPO.register(ModuleType.Histogram)
def fatnjets(events, params, analyzer):
    """Basic information about individual jets"""
    gj = events.good_fat_jets
    analyzer.H(f"h_njet", CommonAxes.nj_axis, ak.num(gj))


@MODULE_REPO.register(ModuleType.Histogram)
def topfatjet_plots(events, params, analyzer):
    mask = ak.num(events.good_fatjets, axis=1) > 0
    top_fj = events.good_fatjets[mask][:, 0]
    analyzer.H(
        f"ak8_pt",
        makeAxis(100, 0, 1000, "$p_{T}$", unit="GeV"),
        top_fj.pt,
        description="pt of the leading ak8 jet",
        mask=mask,
    )

    analyzer.H(
        f"ak8_pt_vs_sdmass",
        [
            makeAxis(100, 0, 1000, "$p_{T}$", unit="GeV"),
            makeAxis(30, 0, 300, "$m_{\\text{SD}}$", unit="GeV"),
        ],
        [top_fj.pt, top_fj.msoftdrop],
        description="2D plot of pt and softdrop mass",
        mask=mask,
    )


@MODULE_REPO.register(ModuleType.Histogram)
def goodjet_ht(events, params, analyzer):
    analyzer.H(
        f"HT",
        makeAxis(120, 0, 3000, "HT", unit="GeV"),
        events.HT,
        description="Sum of $p_T$ of good AK4 jets.",
    )


@MODULE_REPO.register(ModuleType.Histogram)
def trifecta_plot(events, params, analyzer):
    analyzer.H(
        f"ht_ak8_pt_vs_sdmass",
        [
            makeAxis(120, 0, 3000, "HT", unit="GeV"),
            makeAxis(100, 0, 1000, "$p_{T}$", unit="GeV"),
            makeAxis(30, 0, 300, "$m_{\\text{SD}}$", unit="GeV"),
        ],
        [events.HT, events.good_fatjets[:, 0].pt, events.good_fatjets[:, 0].msoftdrop],
        description="3D plot of ht and ak8 pt and softdrop mass",
    )


@MODULE_REPO.register(ModuleType.Histogram)
def jet_kinematics(events, params, analyzer):
    """Basic information about individual jets"""

    gj = events.good_jets
    for i in range(0, 4):
        mask = ak.num(gj, axis=1) > i
        masked_jets = gj[mask]

        analyzer.H(
            rf"pt_{i+1}",
            makeAxis(100, 0, 1000, f"$p_{{T, {i+1}}}$", unit="GeV"),
            gj[:, i].pt,
            description=f"$p_T$ of jet {i+1} ",
        )
        analyzer.H(
            f"eta_{i+1}",
            makeAxis(50, -5, 5, f"$\eta_{{{i+1}}}$"),
            gj[:, i].eta,
            description=f"$\eta$ of jet {i+1}",
        )
        analyzer.H(
            f"phi_{i+1}",
            makeAxis(50, -5, 5, f"$\phi_{{{i+1}}}$"),
            gj[:, i].phi,
            description=f"$\phi$ of jet {i+1}",
        )
        htratio = masked_jets[:, i].pt / events.HT[mask]
        analyzer.H(
            f"pt_ht_ratio_{i}",
            hist.axis.Regular(50, 0, 1, name="pt_o_ht", label=r"$\frac{p_{T}}{HT}$"),
            htratio,
            mask=mask,
            description=rf"Ratio of jet {i} $p_T$ to event HT",
        )

@MODULE_REPO.register(ModuleType.Histogram)
def jet_kinematics_detailed(events, params, analyzer):
    """Basic information about individual jets"""

    gj = events.good_jets
    for i in range(0, 4):
        mask = ak.num(gj, axis=1) > i

        analyzer.H(
            rf"pt_{i+1}",
            makeAxis(300, 0, 3000, f"$p_{{T, {i+1}}}$", unit="GeV"),
            gj[:, i].pt,
            description=f"$p_T$ of jet {i+1} ",
        )
        analyzer.H(
            f"eta_{i+1}",
            makeAxis(50, -5, 5, f"$\eta_{{{i+1}}}$"),
            gj[:, i].eta,
            description=f"$\eta$ of jet {i+1}",
        )
        analyzer.H(
            f"phi_{i+1}",
            makeAxis(50, -5, 5, f"$\phi_{{{i+1}}}$"),
            gj[:, i].phi,
            description=f"$\phi$ of jet {i+1}",
        )


@MODULE_REPO.register(ModuleType.Histogram)
def jet_combo_kinematics(events, params, analyzer):
    """Kinematic information about combinations of jets, such as the naive chargino reconstruction algorithms."""

    gj = events.good_jets
    jet_combos = [
        ((0, 4), "4"),
        ((0, 3), "3,\\text{compressed}"),
        ((1, 4), "3,\\text{uncompressed"),
    ]
    co = lambda x: it.combinations(x, 2)
    masses = {}
    for (i, j), title in jet_combos:
        jets = gj[:, i:j].sum()
        masses[(i, j)] = jets.mass
        analyzer.H(
            f"m{i+1}{j}_pt",
            makeAxis(
                100,
                0,
                1000,
                f"$p_T ( \sum_{{n={i+1}}}^{{{j}}} jet_{{n}})$ ",
                unit="GeV",
            ),
            jets.pt,
            description=f"$p_T$ of the sum of jets {i+1} to {j}",
        )
        analyzer.H(
            f"m{i+1}{j}_eta",
            makeAxis(20, -5, 5, f"$\eta ( \\sum_{{n={i+1}}}^{{{j}}} ) jet_{{n}}$"),
            jets.eta,
            description=rf"$\eta$ of the sum of jets {i+1} to {j}",
        )

        mtitle = 4 if j - i == 4 else 3
        analyzer.H(
            rf"m{i+1}{j}_m",
            makeAxis(300, 0, 3000, f"$m_{{{title}}}$", unit="GeV"),
            jets.mass,
            description=rf"Mass of the sum of jets {i+1} to {j}",
        )

    for (p1, title1), (p2, title2) in [
        (
            ((0, 3), "3,\\text{compressed}"),
            ((0, 4), "4"),
        ),
        (
            ((1, 4), "3,\\text{uncompressed"),
            ((0, 4), "4"),
        ),
    ]:
        p1_1, p1_2 = p1
        p2_1, p2_2 = p2
        analyzer.H(
            f"m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}",
            [
                makeAxis(200, 0, 3000, rf"$m_{{{title2}}}$", unit="GeV"),
                makeAxis(200, 0, 3000, rf"$m_{{{title1}}}$", unit="GeV"),
            ],
            [masses[p2], masses[p1]],
        )

        analyzer.H(
            f"ratio_m{p1_1+1}{p1_2}_vs_m{p2_1+1}{p2_2}",
            [
                makeAxis(200, 0, 3000, rf"$m_{{{title2}}}$", unit="GeV"),
                makeAxis(
                    200,
                    0,
                    1,
                    rf"$\frac{{m_{{ {title1} }} }}{{ m_{{ {title2} }} }}$",
                ),
            ],
            [masses[p2], masses[p1] / masses[p2]],
        )


@MODULE_REPO.register(ModuleType.Histogram)
def jet_relative_angles(events, params, analyzer):
    gj = events.good_jets
    masks = {}
    for i, j in list(x for x in it.combinations(range(0, 4), 2) if x[0] != x[1]):
        mask = ak.num(gj, axis=1) > max(i, j)
        masked_jets = gj[mask]
        d_eta = masked_jets[:, i].eta - masked_jets[:, j].eta
        d_r = abs(masked_jets[:, i].delta_r(masked_jets[:, j]))
        d_phi = masked_jets[:, i].phi - masked_jets[:, j].phi
        masks[(i, j)] = mask
        analyzer.H(
            rf"d_eta_{i+1}_{j+1}",
            makeAxis(50, -6, 6, f"$\Delta \eta_{{{i+1}{j+1}}}$"),
            d_eta,
            mask=mask,
            description=rf"$\Delta \eta$ between jets {i+1} and {j+1}",
        )
        analyzer.H(
            f"d_phi_{i+1}_{j+1}",
            makeAxis(50, -6, 6, f"$\Delta \phi_{{{i+1}{j+1}}}$"),
            d_phi,
            mask=mask,
            description=rf"$\Delta \phi$ between jets {i+1} and {j+1}",
        )
        analyzer.H(
            f"d_r_{i+1}_{j+1}",
            makeAxis(25, 0, 5, f"$|\Delta R_{{{i+1}{j+1}}}|$"),
            d_r,
            mask=mask,
            description=rf"$\Delta R$ between jets {i+1} and {j+1}",
        )

@MODULE_REPO.register(ModuleType.Histogram)
def dijet_hists_exo20008(events, params, analyzer):
    """Histograms for dijet analysis"""
    dijet = events.wide_jet0 + events.wide_jet1

    analyzer.H(
        "dijet_mass",
        makeAxis(100, 1530, 10000, "$m_{jj}$", unit="GeV"),
        dijet.mass,
        description="Mass of the dijet system",
    )

    analyzer.H(
        "wide_jet1_pt",
        makeAxis(125, 0, 5000, "$p_{T,1}$", unit="GeV"),
        events.wide_jet0.pt,
        description="Transverse momentum of the first wide jet", 
    )
    analyzer.H(
        "wide_jet2_pt",
        makeAxis(125, 0, 5000, "$p_{T,2}$", unit="GeV"),
        events.wide_jet1.pt,
        description="Transverse momentum of the second wide jet",
    )
    analyzer.H(
        "wide_jet1_eta",
        makeAxis(50, -3, 3, "$\eta_{1}$"),
        events.wide_jet0.eta,
        description="Pseudorapidity of the first wide jet",
    )
    analyzer.H(
        "wide_jet2_eta",
        makeAxis(50, -3, 3, "$\eta_{2}$"),
        events.wide_jet1.eta,
        description="Pseudorapidity of the second wide jet",
    )
    analyzer.H(
        "wide_jet1_phi",
        makeAxis(50, -3.2, 3.2, "$\phi_{1}$"),
        events.wide_jet0.phi,
        description="Azimuthal angle of the first wide jet",
    )
    analyzer.H(
        "wide_jet2_phi",
        makeAxis(50, -3.2, 3.2, "$\phi_{2}$"),
        events.wide_jet1.phi,
        description="Azimuthal angle of the second wide jet",
    )
    analyzer.H(
        "wide_jet1_mass",
        makeAxis(100, 0, 5000, "$m_{1}$", unit="GeV"),
        events.wide_jet0.mass,
        description="Mass of the first wide jet",
    )
    analyzer.H(
        "wide_jet2_mass",
        makeAxis(100, 0, 5000, "$m_{2}$", unit="GeV"),
        events.wide_jet1.mass,
        description="Mass of the second wide jet",
    )
    analyzer.H(
        "dijet_eta",
        makeAxis(50, 0, 5, "$|\Delta \eta_{jj}|$"),
        abs(events.wide_jet0.eta - events.wide_jet1.eta),
        description="Pseudorapidity difference between the two wide jets",
    )
    analyzer.H(
        "dijet_phi",
        makeAxis(50, -3.2, 3.2, "$\Delta \phi_{jj}$"),
        events.wide_jet0.phi - events.wide_jet1.phi,
        description="Azimuthal angle difference between the two wide jets",
    )
    analyzer.H(
        "dijet_delta_r",
        makeAxis(50, 0, 5, "$\Delta R_{jj}$"),
        events.wide_jet0.delta_r(events.wide_jet1),
        description="Delta R between the two wide jets",
    )
    analyzer.H(
        "dijet_pt",
        makeAxis(100, 0, 5000, "$p_{T, jj}$", unit="GeV"),
        dijet.pt,
        description="Transverse momentum of the dijet system",
    )
    #analyzer.H(
    #    "widejet1_DeepJet",
    #    makeAxis(50, 0, 1, "DeepJet"),
    #    events.wide_jet0.btagDeepFlavB,
    #    description="DeepJet b-tagging score for the first wide jet",
    #)
    #analyzer.H(
    #    "widejet2_DeepJet",
    #    makeAxis(50, 0, 1, "DeepJet"),
    #    events.wide_jet1.btagDeepFlavB,
    #    description="DeepJet b-tagging score for the second wide jet",
    #)
    #analyzer.H(
    #    "dijet_DeepJet",
    #    makeAxis(50, 0, 1, "DeepJet"),
    #    dijet.btagDeepFlavB,
    #    description="DeepJet b-tagging score for the dijet system",
    #)


@MODULE_REPO.register(ModuleType.Histogram)
def dijet_hists(events, params, analyzer):
    """Histograms for dijet analysis"""
    good_fat_jets = events.good_fat_jets
    padded_fatjets = ak.pad_none(good_fat_jets, 2, axis=1)[:,:2]
    sorted_good_fat_jets = padded_fatjets[ak.argsort(padded_fatjets.msoftdrop,ascending=False)]
    fatjet_1 = sorted_good_fat_jets[:, 0]
    fatjet_2 = sorted_good_fat_jets[:, 1]

    dijet=fatjet_1+fatjet_2

    analyzer.H(
        "dijet_mass",
        makeAxis(100, 0, 5000, "$m_{jj}$", unit="GeV"),
        dijet.mass,
        description="Mass of the dijet system",
    )
    analyzer.H(
        "dijet_eta",
        makeAxis(50, 0, 5, "$|\Delta \eta_{jj}|$"),
        abs(fatjet_1.eta - fatjet_2.eta),
        description="Pseudorapidity difference between the two wide jets",
    )
    analyzer.H(
        "dijet_phi",
        makeAxis(50, -3.2, 3.2, "$\Delta \phi_{jj}$"),
        abs(fatjet_1.phi - fatjet_2.phi),
        description="Azimuthal angle difference between the two wide jets",
    )
    analyzer.H(
        "dijet_delta_r",
        makeAxis(50, 0, 5, "$\Delta R_{jj}$"),
        fatjet_1.delta_r(fatjet_2),
        description="Delta R between the two wide jets",
    )
    analyzer.H(
        "dijet_pt",
        makeAxis(100, 0, 5000, "$p_{T, jj}$", unit="GeV"),
        dijet.pt,
        description="Transverse momentum of the dijet system",
    )

    analyzer.H(
        "fat_jet1_pt",
        makeAxis(125, 0, 3000, "$p_{T,1}$", unit="GeV"),
        fatjet_1.pt,
        description="Transverse momentum of the first wide jet", 
    )
    analyzer.H(
        "fat_jet2_pt",
        makeAxis(125, 0, 3000, "$p_{T,2}$", unit="GeV"),
        fatjet_2.pt,
        description="Transverse momentum of the second wide jet",
    )

    analyzer.H(
        "fat_jet1_eta",
        makeAxis(50, -3, 3, "$\eta_{1}$"),
        fatjet_1.eta,
        description="Pseudorapidity of the first wide jet",
    )
    analyzer.H(
        "fat_jet2_eta",
        makeAxis(50, -3, 3, "$\eta_{2}$"),
        fatjet_2.eta,
        description="Pseudorapidity of the second wide jet",
    )

    analyzer.H(
        "fat_jet1_phi",
        makeAxis(50, -3.2, 3.2, "$\phi_{1}$"),
        fatjet_1.phi,
        description="Azimuthal angle of the first wide jet",
    )
    analyzer.H(
        "fat_jet2_phi",
        makeAxis(50, -3.2, 3.2, "$\phi_{2}$"),
        fatjet_2.phi,
        description="Azimuthal angle of the second wide jet",
    )

    analyzer.H(
        "fat_jet1_mass",
        makeAxis(100, 0, 2500, "$m_{1}$", unit="GeV"),
        fatjet_1.mass,
        description="Mass of the first fat jet",
    )
    analyzer.H(
        "fat_jet2_mass",
        makeAxis(100, 0, 2500, "$m_{2}$", unit="GeV"),
        fatjet_2.mass,
        description="Mass of the second fat jet",
    )

    analyzer.H(
        "fat_jet1_tau1",
        makeAxis(100,0,1, "Tau1"),
        fatjet_1.tau1,
        description="Nsubjettiness Tau 1 of FatJet1"
    )
    analyzer.H(
        "fat_jet2_tau1",
        makeAxis(100,0,1, "Tau1"),
        fatjet_2.tau1,
        description="Nsubjettiness Tau 1 of FatJet2"
    )

    analyzer.H(
        "fat_jet1_tau2",
        makeAxis(100,0,1, "Tau2"),
        fatjet_1.tau2,
        description="Nsubjettiness Tau 2 of FatJet1"
    )

    analyzer.H(
        "fat_jet2_tau2",
        makeAxis(100,0,1, "Tau2"),
        fatjet_2.tau2,
        description="Nsubjettiness Tau 2 of FatJet2"
    )

    analyzer.H(
        "fat_jet1_tau3",
        makeAxis(100,0,1, "Tau3"),
        fatjet_1.tau3,
        description="Nsubjettiness Tau 3 of FatJet1"
    )
    analyzer.H(
        "fat_jet2_tau3",
        makeAxis(100,0,1, "Tau3"),
        fatjet_2.tau3,
        description="Nsubjettiness Tau 3 of FatJet2"
    )

    analyzer.H(
        "fat_jet1_msoftdrop",
        makeAxis(100,0,1000, "$m_{SD}$"),
        fatjet_1.msoftdrop,
        description="FatJet1 Softdrop Mass"
    ) 
    analyzer.H(
        "fat_jet2_msoftdrop",
        makeAxis(100,0,1000, "$m_{SD}$"),
        fatjet_2.msoftdrop,
        description="FatJet2 Softdrop Mass"
    ) 

    analyzer.H(
        "fat_jet1_tau31",
        makeAxis(100,0,1, "Tau31"),
        fatjet_1.tau3/fatjet_1.tau1,
        description="Nsubjettiness Tau 3/Tau 1 of FatJet1"
    )
    analyzer.H(
        "fat_jet2_tau31",
        makeAxis(100,0,1, "Tau31"),
        fatjet_2.tau3/fatjet_2.tau1,
        description="Nsubjettiness Tau 3/Tau 1 of FatJet2"
    )

    analyzer.H(
        "fat_jet1_tau32",
        makeAxis(100,0,1, "Tau32"),
        fatjet_1.tau3/fatjet_1.tau2,
        description="Nsubjettiness Tau 3/Tau 2 of FatJet1"
    )
    analyzer.H(
        "fat_jet2_tau32",
        makeAxis(100,0,1, "Tau32"),
        fatjet_2.tau3/fatjet_2.tau2,
        description="Nsubjettiness Tau 3/Tau 2 of FatJet2"
    )

    analyzer.H(
        "fat_jet1_tau21",
        makeAxis(100,0,1, "Tau21"),
        fatjet_1.tau2/fatjet_1.tau1,
        description="Nsubjettiness Tau 2/Tau 1 of FatJet1",
    )
    analyzer.H(
        "fat_jet2_tau21",
        makeAxis(100,0,1, "Tau21"),
        fatjet_2.tau2/fatjet_2.tau1,
        description="Nsubjettiness Tau 2/Tau 1 of FatJet2",
    )

    analyzer.H(
        "fat_jet1_bscore",
        makeAxis(100,0,1,"DeepB score"),
        fatjet_1.btagDeepB,
        description="DeepB Score of FatJet1",
    )
    analyzer.H(
        "fat_jet2_bscore",
        makeAxis(100,0,1,"DeepB score"),
        fatjet_2.btagDeepB,
        description="DeepB Score of FatJet2",
    )