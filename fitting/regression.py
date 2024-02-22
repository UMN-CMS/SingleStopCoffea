import sys
import pickle as pkl
import hist
from analyzer.core import AnalysisResult
from analyzer.datasets import SampleManager
import contextlib
import math
import torch
import gpytorch
from torch.masked import masked_tensor, as_masked_tensor
from collections import namedtuple
from .models import ExactGPModel, ExactProjGPModel, ExactAnyKernelModel
from rich.progress import track


DataValues = namedtuple("DataValues", "inputs outputs variances edges")


class SimpleTransformer:
    def __init__(self, slope, intercept):
        self.slope = slope
        self.intercept = intercept

    def transformData(self, data):
        return (data - self.intercept) / slope

    def iTransformData(self, data):
        return (data * slope) + intercept

    def transformVariances(self, v):
        return v / self.slope**2

    def transformVariances(self, v):
        return v * self.slope**2


def preprocessHistograms(background_hist, mask_region):
    def make_mask(x1, x2):
        if mask_region is None:
            return (
                torch.full_like(x1, False, dtype=torch.bool),
                torch.full_like(x2, False, dtype=torch.bool),
            )
        else:
            a = (x1 > mask_region[0][0]) & (x1 < mask_region[0][1])
            b = (x2 > mask_region[1][0]) & (x2 < mask_region[1][1])
            return a, b

    edges_x1 = torch.from_numpy(background_hist.axes[0].edges)
    edges_x2 = torch.from_numpy(background_hist.axes[1].edges)
    centers_x1 = torch.diff(edges_x1) / 2 + edges_x1[:-1]
    centers_x2 = torch.diff(edges_x2) / 2 + edges_x2[:-1]

    bin_values = torch.from_numpy(background_hist.values()).T
    bin_vars = torch.from_numpy(background_hist.variances()).T

    emask_x1, emask_x2 = make_mask(edges_x1, edges_x2)
    centers_grid_x1, centers_grid_x2 = torch.meshgrid(
        centers_x1, centers_x2, indexing="xy"
    )

    cmask_x1, cmask_x2 = make_mask(centers_x1, centers_x2)

    centers_grid = torch.stack((centers_grid_x1, centers_grid_x2), axis=2)
    m1, m2 = make_mask(centers_grid[:, :, 0], centers_grid[:, :, 1])
    centers_mask = m1 & m2
    flat_centers = torch.flatten(centers_grid, end_dim=1)
    flat_bin_values = torch.flatten(bin_values)
    flat_bin_vars = torch.flatten(bin_vars)
    ma = torch.max(flat_centers.T, axis=1).values
    mi = torch.min(flat_centers.T, axis=1).values
    transformed_centers = (flat_centers - mi) / (ma - mi)

    ma = torch.max(flat_bin_values)
    mi = torch.min(flat_bin_values)
    value_scale = ma - mi
    transformed_values = (flat_bin_values) / value_scale
    transformed_vars = flat_bin_vars / value_scale**2

    flat_centers_masked = transformed_centers[torch.flatten(~centers_mask)]
    flat_bin_values_masked = transformed_values[torch.flatten(~centers_mask)]
    flat_bin_vars_masked = transformed_vars[torch.flatten(~centers_mask)]

    train_x = flat_centers_masked
    train_y = flat_bin_values_masked
    train_vars = flat_bin_vars_masked
    test_x = transformed_centers
    test_y = transformed_values
    test_vars = transformed_vars

    return (
        (
            DataValues(
                flat_centers[torch.flatten(~centers_mask)],
                flat_bin_values[torch.flatten(~centers_mask)],
                flat_bin_vars[torch.flatten(~centers_mask)],
                (edges_x1, edges_x2),
            ),
            DataValues(
                flat_centers, flat_bin_values, flat_bin_vars, (edges_x1, edges_x2)
            ),
        ),
        (
            DataValues(train_x, train_y, train_vars, (edges_x1, edges_x2)),
            DataValues(test_x, test_y, test_vars, (edges_x1, edges_x2)),
        ),
        centers_mask,
        value_scale,
    )


def createModel(train_data, kernel=None, model_maker=None):
    # v = torch.maximum(train_data.variances, torch.tensor(0.00001))
    v = train_data.variances

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=v, learn_additional_noise=False
    )
    if model_maker is None:
        model_maker = ExactAnyKernelModel

    if kernel:
        model = model_maker(
            train_data.inputs, train_data.outputs, likelihood, kernel=kernel
        )
    else:
        model = model_maker(train_data.inputs, train_data.outputs, likelihood)
    return model, likelihood


def optimizeHyperparams(model, likelihood, train_data, iterations=100, bar=True):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.05)
    # optimizer = torch.optim.Adam([
    #         {'params': model.covar_module.base_kernel.raw_lengthscale, 'lr': 20},
    #         {'params': model.covar_module.raw_outputscale, 'lr': 5e6},
    #         #{'params': model.proj_mat, 'lr': 0.25},
    #         {'params': model.mean_module.parameters(), 'lr': 1e4},
    #        # {"params" : model.likelihood.parameters(), 'lr' : 1e2}
    # ])

    mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

    if bar:
        iterator = track(range(iterations), description="Optimizing Hyperparams...")
    else:
        iterator = range(iterations)

    for i in iterator:
        optimizer.zero_grad()
        output = model(train_data.inputs)
        loss = -mll(output, train_data.outputs)
        loss.backward()
        # if not (i + 1) % (iterations / 10) or i == 0:
        #    print(
        #        "Iter {:d} - Loss: {:0.3f}   lengthscale: {:0.4f},{:0.4f}  scale: {:0.3f}  mean:{:0.3f} noise:{:0.3f} ".format(
        #            i + 1,
        #            loss.item(),
        #            model.covar_module.base_kernel.lengthscale.squeeze()[0],
        #            model.covar_module.base_kernel.lengthscale.squeeze()[1],
        #            model.covar_module.outputscale.item(),
        #            model.mean_module.constant.item(),
        #            likelihood.noise.mean(),
        #        )
        #    )
        #    # print(model.proj_mat)
        optimizer.step()
    return model, likelihood


def getPrediction(model, likelihood, test_data):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_computations():
        observed_pred = likelihood(model(test_data.inputs), noise=test_data.variances)
    return observed_pred
