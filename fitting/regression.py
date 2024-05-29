import contextlib
from collections import namedtuple
from dataclasses import dataclass

import gpytorch
import torch
from rich.progress import Progress
from .models import ExactAnyKernelModel

DataValues = namedtuple("DataValues", "X Y V E")


@dataclass
class DataValues:
    X: torch.Tensor
    Y: torch.Tensor
    V: torch.Tensor
    E: torch.Tensor

    def toGpu(self):
        return DataValues(self.X.cuda(), self.Y.cuda(), self.V.cuda(), self.E)

    def fromGpu(self):
        return DataValues(self.X.cpu(), self.Y.cpu(), self.V.cpu(), self.E)


def makeRegressionData(
    histogram, mask_function=None, exclude_less=None, get_mask=False
):
    if mask_function is None:
        mask_function = lambda x1, x2: (
            torch.full_like(x1, False, dtype=torch.bool),
            torch.full_like(x2, False, dtype=torch.bool),
        )

    edges_x1 = torch.from_numpy(histogram.axes[0].edges)
    edges_x2 = torch.from_numpy(histogram.axes[1].edges)

    centers_x1 = torch.diff(edges_x1) / 2 + edges_x1[:-1]
    centers_x2 = torch.diff(edges_x2) / 2 + edges_x2[:-1]

    bin_values = torch.from_numpy(histogram.values()).T
    bin_vars = torch.from_numpy(histogram.variances()).T
    centers_grid_x1, centers_grid_x2 = torch.meshgrid(
        centers_x1, centers_x2, indexing="xy"
    )
    if exclude_less:
        domain_mask = bin_values < exclude_less
    else:
        domain_mask = torch.full_like(bin_values, False, dtype=torch.bool)

    centers_grid = torch.stack((centers_grid_x1, centers_grid_x2), axis=2)
    m1, m2 = mask_function(centers_grid[:, :, 0], centers_grid[:, :, 1])
    centers_mask = (m1 | domain_mask) & (m2 | domain_mask)
    flat_centers = torch.flatten(centers_grid, end_dim=1)
    flat_bin_values = torch.flatten(bin_values)
    flat_bin_vars = torch.flatten(bin_vars)
    ret = DataValues(
        flat_centers[torch.flatten(~centers_mask)],
        flat_bin_values[torch.flatten(~centers_mask)],
        flat_bin_vars[torch.flatten(~centers_mask)],
        (edges_x1, edges_x2),
    )
    if get_mask:
        return ret, torch.flatten(~centers_mask)
    else:
        return ret


def createModel(train_data, kernel=None, model_maker=None, learn_noise=False, **kwargs):
    v = train_data.V

    likelihood = gpytorch.likelihoods.FixedNoiseGaussianLikelihood(
        noise=v,
        learn_additional_noise=learn_noise,
    )
    if model_maker is None:
        model_maker = ExactAnyKernelModel

    if kernel:
        model = model_maker(
            train_data.X, train_data.Y, likelihood, kernel=kernel, **kwargs
        )
    else:
        model = model_maker(train_data.X, train_data.Y, likelihood, **kwargs)
    return model, likelihood

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def optimizeHyperparams(
    model,
    likelihood,
    train_data,
    iterations=100,
    bar=True,
    lr=0.05,
    get_evidence=False,
    mll=None,
):
    model.train()
    likelihood.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    if mll is None:
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)


    print(f"step_size={iterations//3}")
    scheduler = torch.optim.lr_scheduler.StepLR(
       optimizer, step_size=iterations//3, gamma=0.1
    )
    #scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min")

    context = Progress() if bar else contextlib.nullcontext()
    evidence = None

    with context as progress:
        if bar:
            task1 = progress.add_task("[red]Optimizing...", total=iterations)
        for i in range(iterations):
            optimizer.zero_grad()
            output = model(train_data.X)
            loss = -mll(output, train_data.Y)

            loss.backward()
            optimizer.step()
            #scheduler.step(loss)
            scheduler.step()

            #slr = get_lr(optimizer)
            slr= scheduler.get_last_lr()[0]

            #if slr < 1e-3 * lr:
            #    print(
            #        f"Iter {i} (lr={slr}): Loss = {round(loss.item(),4)}"
            #    )
            #    evidence = float(loss.item())
            #    break
            if bar:
                progress.update(
                    task1,
                    advance=1,
                    description=f"[red]Optimizing(Loss is {round(loss.item(),3)})...",
                )
                progress.refresh()
            else:
                if (i % (iterations // 10) == 0) or i == iterations - 1:
                    print(
                        f"Iter {i} (lr={slr}): Loss = {round(loss.item(),4)}"
                    )
                    evidence = float(loss.item())
                    pass

    if get_evidence:
        return model, likelihood, evidence
    else:
        return model, likelihood


def getPrediction(model, likelihood, test_data):
    model.eval()
    likelihood.eval()
    with torch.no_grad(), gpytorch.settings.fast_pred_var():
        observed_pred = model(test_data.X)
    return observed_pred


def getBlindedMask(inputs, pred_mean, test_mean, test_var, mask_func):
    imask_x, imask_y = mask_func(inputs[:, 0], inputs[:, 1])
    mask = imask_x & imask_y
    return mask
    pred_mean = pred_mean[mask]
    test_mean = test_mean[mask]
    test_var = test_var[mask]
    num = torch.count_nonzero(mask)
    return torch.sum((test_mean - pred_mean) ** 2 / test_var) / num
