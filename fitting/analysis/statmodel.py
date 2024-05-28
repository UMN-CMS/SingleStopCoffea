from pathlib import Path

import hist
import pyro
import pyro.distributions as dist
import pyro.infer.reparam as pir
import pyro.poutine as poutine
import torch
from analyzer.core import AnalysisResult
from analyzer.datasets import SampleManager
from fitting.high_level import RegressionModel
from fitting.regression import DataValues, makeRegressionData
from fitting.utils import getScaledEigenvecs
from pyro.infer import HMC, MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal
from pyro.optim import Adam


def statModel(bkg_mean, bkg_transform, signal_dist, observed=None):
    r = pyro.sample("rate", dist.Uniform(-20, 20))
    with pyro.plate("background_variations", bkg_transform.shape[1]):
        b = pyro.sample("raw_variations", dist.Normal(0, 1))
    background = bkg_mean + bkg_transform @ b
    obs_hist = (r * signal_dist) + background
    with pyro.plate("bins", bkg_mean.shape[0]):
        return pyro.sample(
            "observed", dist.Poisson(torch.clamp(obs_hist, 1)), obs=observed
        )


def runMCMC(model, *args, **kwargs):
    nuts_kernel = NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        jit_compile=True,
    )
    mcmc = MCMC(nuts_kernel, num_samples=800, warmup_steps=400,num_chains=1)
    mcmc.run(*args, **kwargs)
    return mcmc


def main():
    import sys

    sample_manager = SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")

    reg_model = torch.load(Path(sys.argv[1]))
    pred_dist = reg_model.posterior_dist
    real = reg_model.test_data.Y
    domain_mask = reg_model.domain_mask

    sig_res = AnalysisResult.fromFile(sys.argv[2])
    sighists = sig_res.getMergedHistograms(sample_manager)
    sig_hist = sighists["ratio_m14_vs_m24"][
        "signal_312_1500_900",
        hist.loc(1000) : hist.loc(3000),
        hist.loc(0.35) : hist.loc(1),
    ]
    signal_data = makeRegressionData(sig_hist)
    signal_data = DataValues(
        signal_data.X[domain_mask],
        signal_data.Y[domain_mask],
        signal_data.V[domain_mask],
        signal_data.E,
    )
    signal_dist = signal_data.Y
    obs = real + 1 * signal_dist
    evars = getScaledEigenvecs(pred_dist.covariance_matrix)
    mcmc = runMCMC(statModel, pred_dist.mean, evars, signal_dist, observed=obs)
    mcmc.summary()


if __name__ == "__main__":
    main()
