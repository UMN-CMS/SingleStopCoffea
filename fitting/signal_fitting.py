import pyro
import pyro.distributions as dist
import pyro.infer.reparam as pir
import torch
from pyro.infer import HMC, MCMC, NUTS, SVI, Predictive, Trace_ELBO
from pyro.infer.autoguide import AutoDelta, AutoDiagonalNormal, AutoMultivariateNormal


def simpleSignalModel(signal_locs, signal_data=None):
    A = pyro.sample("A", dist.Uniform(0, 100000))
    # mu = pyro.param("mu", dist.Uniform(torch.min(signal_locs,axis=0), torch.max(signal_locs,axis=0)))
    #lmin = torch.min(signal_locs, axis=0).values
    lmax = torch.max(signal_locs, axis=0).values

    #signal_locs = signal_locs / lmax

    u = dist.Uniform(torch.Tensor([0, 0]), lmax).to_event(1)
    mu = pyro.sample("mu", u)

    m = 1e-10
    evals = pyro.sample(
        "evals",
        dist.Uniform(
            torch.Tensor([m, m]),  lmax
        ).to_event(1),
    )
    rot =  pyro.sample( "rot", dist.Uniform(0,6.28))
    #cvp = pyro.sample(
    #    "cvp",
    #    dist.Uniform(
    #        torch.Tensor([[m, m], [m, m]]), torch.tensor([[3.0, 3.0], [3.0, 3.0]])
    #    ).to_event(2),
    #)
    s = torch.sin(rot)
    c = torch.cos(rot)
    r = torch.stack([torch.stack([c, -s]),
                       torch.stack([s, c])])

    cm = r @ torch.diag(evals) @ r.T
    nexp = A * torch.exp(-1 / 2 * torch.einsum("ij,jk,ik->i", signal_locs - mu,  cm, signal_locs - mu))
    nexp = torch.round(nexp)
    nexp = torch.clamp(nexp,1)
    #print(nexp)
    m = nexp >= 1
    #print(torch.sum(m))
    with pyro.plate("data", torch.sum(m)):
        if signal_data:
            sd = signal_data[m]
        else:
            sd =signal_data
        obs = pyro.sample("sig_observation", dist.Poisson(nexp[m]), obs=sd)

    return obs


def simpleSignalModelParam(signal_locs, signal_data=None):
    A = pyro.param("A", dist.Uniform(0, 100000))
    lmax = torch.max(signal_locs, axis=0).values
    signal_locs = signal_locs / lmax
    u = dist.Uniform(torch.Tensor([0, 0]), torch.Tensor([1, 1])).to_event(1)
    mu = pyro.param("mu", torch.Tensor([0.5, 0.5]))
    cvp = pyro.param("cvp", torch.Tensor([[1, 1], [1, 1]]))
    cm = cvp @ cvp.T
    nexp = A * torch.exp(-1 / 2 * torch.einsum("ij,jk,ik->i", signal_locs - mu,  cm, signal_locs - mu))
    nexp = torch.clamp(torch.round(nexp), 1)
    with pyro.plate("data", signal_locs.shape[0]):
        obs = pyro.sample("sig_observation", dist.Poisson(nexp), obs=signal_data)
    return obs
