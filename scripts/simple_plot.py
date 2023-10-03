import sys

sys.path.append(".")
from analyzer.plotting.simple_plot import Plotter
import warnings

warnings.filterwarnings("ignore", message=r".*Removed bins.*")

backgrounds = ["Skim_QCDInclusive2018"]
compressed = [f"signal_312_{p}" for p in ("2000_1900", "1200_1100", "1500_1400")]
uncompressed = [f"signal_312_{p}" for p in ("2000_1400", "1200_400", "1500_900")]
both = compressed + uncompressed
representative = [f"signal_312_{p}" for p in ("2000_1900", "1200_400", "1500_900")]


plotter = Plotter("signalhists.pkl", "figures", default_backgrounds=backgrounds)


plotter("m14_vs_m13", compressed, sig_style="profile", add_label="Compressed")
plotter("h_njet", compressed, add_name="compressed")
plotter("h_njet", uncompressed, add_name="uncompressed")
plotter("m14_vs_m3_top_3_no_lead_b", uncompressed, normalize=True)
plotter("m14_vs_m3_top_2_plus_lead_b", compressed, normalize=True)
plotter("medium_bdr", compressed, add_name="compressed")
plotter("medium_bdr", uncompressed, add_name="uncompressed")
plotter("medium_bb_eta", compressed, add_name="compressed")
plotter("medium_bb_eta", uncompressed, add_name="uncompressed")
plotter("medium_bb_phi", compressed, add_name="compressed")
plotter("medium_bb_phi", uncompressed, add_name="uncompressed")


plotter(
    "chi_b_dr",
    representative,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)
plotter(
    "chi_b_eta",
    representative,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)
plotter(
    "chi_b_phi",
    representative,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter(
    "chi_b_jet_idx",
    representative,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)
plotter(
    "stop_b_jet_idx",
    representative,
    [],
    normalize=True,
    scale="linear",
    sig_style="hist",
)

plotter("HT", representative)


plotter("m3_top_3_no_lead_b", uncompressed)
plotter("m3_top_2_plus_lead_b", compressed)


plotter("ratio_m14_vs_m3_top_3_no_lead_b", uncompressed, normalize=True)
plotter("ratio_m14_vs_m3_top_2_plus_lead_b", compressed, normalize=True)

plotter("lead_medium_bjet_ordinality", compressed, add_name="compressed")
plotter("lead_medium_bjet_ordinality", uncompressed, add_name="uncompressed")

plotter("sublead_medium_bjet_ordinality", compressed, add_name="compressed")
plotter("sublead_medium_bjet_ordinality", uncompressed, add_name="uncompressed")

plotter("num_top_3_jets_matched_chi_children", compressed, [], add_label="Compressed")
plotter("num_top_4_jets_matched_stop_children", representative, [])
plotter(
    "num_sub_3_jets_matched_chi_children", uncompressed, [], add_label="Uncompressed"
)

plotter(
    "mstop_gen_matched",
    compressed + uncompressed,
    [],
    normalize=True,
    scale="linear",
)
plotter(
    "mchi_gen_matched",
    compressed + uncompressed,
    [],
    normalize=True,
    scale="linear",
)

plotter("m14_vs_m24", uncompressed)
plotter("m14_vs_m13", compressed)
plotter("ratio_m14_vs_m24", uncompressed)
plotter("ratio_m14_vs_m13", compressed)

plotter("pt_1", representative)
