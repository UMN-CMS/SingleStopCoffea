from pathlib import Path

import hist
import jax
from jax import random
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist
import numpyro.infer.reparam as pir
from numpyro.infer import HMC, MCMC, NUTS, SVI, Predictive, Trace_ELBO
import torch
from analyzer.core import AnalysisResult
from analyzer.datasets import SampleManager
from fitting.high_level import RegressionModel,SignalData
from fitting.regression import DataValues, makeRegressionData
from fitting.utils import getScaledEigenvecs


def statModel(bkg_mean, bkg_transform, signal_dist, observed=None):
    r = numpyro.sample("rate", dist.Uniform(-20, 20))
    with numpyro.plate("background_variations", bkg_transform.shape[1]):
        b = numpyro.sample("raw_variations", dist.Normal(0, 1))
    background = bkg_mean + bkg_transform @ b
    obs_hist = (r * signal_dist) + background
    with numpyro.plate("bins", bkg_mean.shape[0]):
        return numpyro.sample(
            "observed", dist.Poisson(jnp.clip(obs_hist, 1)), obs=observed
        )


def runMCMC(model, *args, **kwargs):
    rng_key = random.PRNGKey(0)
    nuts_kernel = NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    mcmc = MCMC(nuts_kernel, num_samples=800, num_warmup=200,num_chains=1)
    mcmc.run(rng_key, *args, **kwargs)
    return mcmc


def runMCMCOnDataset(signal_data, regression_data, obs):
    dm = regression_data.domain_mask
    signal_dist = signal_data.signal_data.Y[dm]

    sX = signal_data.signal_data.X[dm]
    assert(torch.allclose(sX, regression_data.test_data.X))
    pred_dist = regression_data.posterior_dist
    evars = getScaledEigenvecs(pred_dist.covariance_matrix)

    s = signal_dist.numpy()
    o =  obs.numpy()
    ev = evars.numpy()
    m = pred_dist.mean.numpy()

    mcmc = runMCMC(statModel, m, ev, s, observed=o)
    return mcmc


def main():
    import sys

    sample_manager = SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")

    reg_model = torch.load(Path(sys.argv[1]))
    sig_data = torch.load(Path(sys.argv[2]))
    sd = sig_data.signal_data.Y[reg_model.domain_mask]
    obs = reg_model.test_data.Y + 0 * sd
    obs = reg_model.posterior_dist.mean + 0 * sd
    mcmc = runMCMCOnDataset(sig_data, reg_model, obs )
    mcmc.print_summary()


if __name__ == "__main__":
    main()
