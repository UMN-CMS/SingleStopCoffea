import concurrent.futures
import pickle
from pathlib import Path

import hist
import jax
import jax.numpy as jnp
import numpyro
import numpyro.distributions as ndist

from pyro.infer import Predictive
import pyro.infer  as pinf
import pyro.distributions as pdist

from jax import random
from numpyro.infer as ninf

import torch

from analyzer.core import AnalysisResult
from analyzer.datasets import SampleManager

from fitting.high_level import RegressionModel, SignalData
from fitting.regression import DataValues, makeRegressionData
from fitting.utils import getScaledEigenvecs

import multiprocessing as mp

def statModelPyro(bkg_mean, bkg_transform, signal_dist, observed=None):
    r = pyro.sample("rate", pdist.Uniform(-20, 20))
    with pyro.plate("background_variations", bkg_transform.shape[1]):
        b = pyro.sample("raw_variations", pdist.Normal(0, 1))
    background = bkg_mean + bkg_transform @ b
    obs_hist = (r * signal_dist) + background
    with pyro.plate("bins", bkg_mean.shape[0]):
        return pyro.sample(
            "observed", pdist.Poisson(torch.clamp(obs_hist, 1)), obs=observed
        )


def runMCMCPyro(model, *args, **kwargs):
    nuts_kernel = pinf.NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
        jit_compile=True,
    )
    mcmc = pinf.MCMC(nuts_kernel, num_samples=800, warmup_steps=400,num_chains=1)
    mcmc.run(*args, **kwargs)
    return mcmc

def runMCMCOnDatasetPyro(signal_data, regression_data, obs):
    dm = regression_data.domain_mask
    signal_dist = signal_data.signal_data.Y[dm]

    sX = signal_data.signal_data.X[dm]
    assert torch.allclose(sX, regression_data.test_data.X)
    pred_dist = regression_data.posterior_dist
    evars = getScaledEigenvecs(pred_dist.covariance_matrix)

    mcmc = runMCMC(statModel, m, ev, s, observed=o)
    return mcmc

def runSVIOnDatasetPyro(signal_data, regression_data, obs):
    dm = regression_data.domain_mask
    signal_dist = signal_data.signal_data.Y[dm]

    sX = signal_data.signal_data.X[dm]
    assert torch.allclose(sX, regression_data.test_data.X)
    pred_dist = regression_data.posterior_dist
    evars = getScaledEigenvecs(pred_dist.covariance_matrix)
    pyro.clear_param_store()

    guide = pyro.infer.autoguide.AutoNormal(pyro_model)

    num_steps = 4000
    initial_lr = 0.1
    gamma = 0.01  # final learning rate will be gamma * initial_lr
    lrd = gamma ** (1 / num_steps)
    adam = pyro.optim.ClippedAdam({'lr': initial_lr, 'lrd': lrd})
    elbo = pyro.infer.Trace_ELBO()

    svi = pyro.infer.SVI(pyro_model, guide, adam, elbo)

    losses = []
    for step in range(num_steps):  # Consider running for more steps.
        loss = svi.step()
        losses.append(loss)
        if step % ( num_steps // 10) == 0:
            print("Elbo loss: {:0.3f}".format(loss))
    predictive = Predictive(conditioned, guide=guide, num_samples=1000)
    return predictive

def statModelNumpyro(bkg_mean, bkg_transform, signal_dist, observed=None):
    r = numpyro.sample("rate", ndist.Uniform(-20, 20))
    with numpyro.plate("background_variations", bkg_transform.shape[1]):
        b = numpyro.sample("raw_variations", ndist.Normal(0, 1))
    background = bkg_mean + bkg_transform @ b
    obs_hist = (r * signal_dist) + background
    with numpyro.plate("bins", bkg_mean.shape[0]):
        return numpyro.sample(
            "observed", ndist.Poisson(jnp.clip(obs_hist, 1)), obs=observed
        )


def runMCMCNumpyro(model, *args, **kwargs):
    rng_key = random.PRNGKey(0)
    nuts_kernel = ninf.NUTS(
        model,
        adapt_step_size=True,
        adapt_mass_matrix=True,
    )
    mcmc = ninf.MCMC(nuts_kernel, num_samples=800, num_warmup=200)
    mcmc.run(rng_key, *args, **kwargs)
    return mcmc


def runMCMCOnDatasetNumpyro(signal_data, regression_data, obs):
    dm = regression_data.domain_mask
    signal_dist = signal_data.signal_data.Y[dm]

    sX = signal_data.signal_data.X[dm]
    assert torch.allclose(sX, regression_data.test_data.X)
    pred_dist = regression_data.posterior_dist
    evars = getScaledEigenvecs(pred_dist.covariance_matrix)

    s = signal_dist.numpy()
    o = obs.numpy()
    ev = evars.numpy()
    m = pred_dist.mean.numpy()

    mcmc = runMCMC(statModel, m, ev, s, observed=o)
    return mcmc


def f(x):
    print(x)
    return runMCMCOnDataset(*x)


def main():
    import sys

    sample_manager = SampleManager()
    sample_manager.loadSamplesFromDirectory("datasets")

    reg_model = torch.load(Path(sys.argv[1]))
    sig_data = torch.load(Path(sys.argv[2]))
    sd = sig_data.signal_data.Y[reg_model.domain_mask]
    obs = reg_model.test_data.Y + 0 * sd
    mcmc = runMCMCOnDataset(sig_data, reg_model, obs)
    mcmc.print_summary()
    pickle.dump(mcmc, open("testmcmc.pkl", "wb"))
    rates = [0, 0.5, 1.0, 4.0]
    if False:
        results = {}
        with concurrent.futures.ProcessPoolExecutor() as executor:
            for r, mc in zip(
                rates,
                executor.map(
                    f,
                    [(sig_data, reg_model, reg_model.test_data.Y + i * sd) for i in rates],
                ),
            ):
                results[r] = mc
        pickle.dump(mc, open("testmcmc.pkl", "wb"))


if __name__ == "__main__":
    print(jax.devices("cpu"))
    #numpyro.set_platform('cpu')
    #jax.default_device = jax.devices("cpu")[0]
    #mp.set_start_method('spawn')
    main()
