from collections import Counter, namedtuple

import analyzer.plotting as plotting
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import torch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from .regression import getPrediction
from .utils import pointsToGrid

Point = namedtuple("Point", "x y")
Square = namedtuple("Square", "bl s")


def makeSquares(points, edges):
    e = torch.meshgrid(edges, indexing="ij")
    e = torch.stack(e, axis=-1)
    e = e[:-1, :-1]
    d = torch.meshgrid(torch.diff(edges[0]), torch.diff(edges[1]), indexing="ij")
    d = torch.stack(d, axis=-1)
    d1 = d.clone()
    d1[..., :] = 0
    d2 = d.clone()
    d2[..., 0] = 0
    d3 = d.clone()
    d3[..., 1] = 0
    e = torch.unsqueeze(e, axis=-2)
    all_diffs = torch.stack((d, d1, d2, d3), axis=-2)
    f = all_diffs + e
    h, _ = pointsToGrid(points, torch.ones_like(points[:, 0]), edges)
    t = f[h.hist > 0]
    return t


def getPolyFromSquares(squares):
    l = list(
        tuple(round(y, 4) for y in x) for x in torch.flatten(squares, 0, 1).tolist()
    )
    boundary_points = list(x for x, y in Counter(l).items() if y % 2 == 1)
    b = torch.tensor(boundary_points)
    center = b.mean(axis=0)
    diffs = b - center
    angles = torch.atan2(diffs[:, 1], diffs[:, 0])
    mask = torch.argsort(angles)
    return b[mask].tolist()


def convexHull(points):
    e = 0.001

    def close(a, b):
        return abs(a - b) < e

    def slope(p1, p2):
        if close(p1[0], p2[0]):
            return float("inf")
        else:
            return 1.0 * ((p2[1] - p1[1]) / (p2[0] - p1[0]))

    def cross(p1, p2, p3):
        ret = ((p2[0] - p1[0]) * (p3[1] - p1[1])) - ((p2[1] - p1[1]) * (p3[0] - p1[0]))
        return ret

    stack = sorted(list(set(points)))
    start = stack.pop()
    stack = sorted(stack, key=lambda x: (slope(x, start), -x[1], x[0]))
    hull = [start]
    for p in stack:
        hull.append(p)
        while len(hull) > 2 and cross(hull[-3], hull[-2], hull[-1]) < 0:
            hull.pop(-2)
    return hull


def plotGaussianProcess(ax, pobj, mask=None):
    mean = pobj.values
    dev = np.sqrt(pobj.variances)
    points = pobj.axes[0].centers
    if mask is not None:
        mean = mean[mask]
        dev = dev[mask]
        points = points[mask]
    ax.plot(points, mean, label="Mean prediction", color="tab:orange")
    ax.fill_between(
        points,
        mean + dev,
        mean - dev,
        alpha=0.3,
        color="tab:orange",
        label=r"Mean$\pm \sigma$",
    )
    return ax


def generatePulls(ax, observed, model, observed_title="", mask=None, domain=None):
    edges, data, variances = observed
    mean, model_variance = model


    model_obj = plotting.PlotObject.fromNumpy((mean, edges), model_variance, mask=mask)
    obs_obj = plotting.PlotObject.fromNumpy(
        (data, edges), variances, title=observed_title, mask=mask
    )

    plotting.drawAsScatter(ax, obs_obj, yerr=True)
    plotGaussianProcess(ax, model_obj, mask=mask)

    ax.tick_params(axis="x", labelbottom=False)
    plotting.addAxesToHist(ax, num_bottom=1, bottom_pad=0)
    # if sig_hist:
    #   drawAs1DHist(ax, PlotObject(sig_hist, "Injected Signal"), yerr=True, fill=None)
    ax.set_yscale("linear")
    if domain:
        ax.set_xrange(domain)

    ab = ax.bottom_axes[0]
    plotting.drawPull(ab, model_obj, obs_obj, hline_list=[-1, 0, 1])
    ab.set_ylim(-3, 3)

    # ls = np.linspace(min_bound, 3000, 2000).reshape(-1, 1)
    # mean_at_pred, upper_at_pred, lower_at_pred, variance_at_pred = getPrediction(
    #    model, torch.from_numpy(ls)
    # )
    # ab.set_ylabel(r"$\frac{obs - pred}{\sigma_{o}}$")

    return ax


def createSlices(
    pred_mean,
    pred_variance,
    test_mean,
    test_variance,
    bin_edges,
    valid,
    slice_dim=1,
    mask_function=None,
    observed_title="",
    domain=None,
    just_window=False,
):
    dim = slice_dim
    num_slices = pred_mean.shape[dim]

    centers = bin_edges[dim][:-1] + torch.diff(bin_edges[dim]) / 2
    c0 = bin_edges[0][:-1] + torch.diff(bin_edges[0]) / 2
    c1 = bin_edges[1][:-1] + torch.diff(bin_edges[1]) / 2
    c = (c0, c1)
    if mask_function:
        m1, m2 = mask_function(*torch.meshgrid(c, indexing="xy"))
        mask = m1 & m2
        mask = mask.T
    else:
        mask = None
    orth_ax = c[dim]
    main_ax = c[1 - dim]
    orth_e = bin_edges[dim]
    main_e = bin_edges[1 - dim]
    for i in range(num_slices):
        val = centers[i]
        if mask_function:
            s_mask = mask.select(dim, i)
            in_win1 = main_e[torch.cat((s_mask, torch.tensor([False])))]
            in_win2 = main_e[torch.cat((torch.tensor([False]), s_mask))]

        if mask_function and len(in_win1) != 0:
            window = [torch.min(in_win1), torch.max(in_win2)]
        else:
            window = None

        if just_window and window is None:
            continue

        fill_mask = valid.select(dim, i)

        slice_pred_mean = pred_mean.select(dim, i)
        slice_pred_var = pred_variance.select(dim, i)

        slice_obs_mean = test_mean.select(dim, i)
        slice_obs_var = test_variance.select(dim, i)
        fig, ax = plt.subplots()
        generatePulls(
            ax,
            (bin_edges[1 - dim], slice_obs_mean, slice_obs_var),
            (slice_pred_mean, slice_pred_var),
            observed_title=observed_title,
            mask=fill_mask,
            domain=domain,
        )

        if window:
            ax.axvline(window[0], color="red", linewidth=0.3, linestyle="-.")
            ax.axvline(window[1], color="red", linewidth=0.3, linestyle="-.")

            ax.bottom_axes[0].axvline(
                window[0], color="red", linewidth=0.3, linestyle="-."
            )
            ax.bottom_axes[0].axvline(
                window[1], color="red", linewidth=0.3, linestyle="-."
            )
        plotting.addEra(ax, "59.83")
        plotting.addPrelim(ax)
        plotting.addText(
            ax,
            0.98,
            0.5,
            f"Val={round(float(val),2)}",
            horizontalalignment="right",
            verticalalignment="bottom",
        )
        ax.bottom_axes[0].set_ylabel(r"$\frac{obs - pred}{\sigma_{o}}$")
        ax.legend()
        yield val, fig, ax


def simpleGrid(ax, edges, inx, iny, cmap="viridis"):
    def addColorbar(ax, vals):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(vals, cax=cax)
        cax.get_yaxis().set_offset_position("left")
        ax.cax = cax

    X, Y = np.meshgrid(*edges)
    z = iny
    Z, filled = pointsToGrid(inx, iny, edges)
    Z = Z.hist.T
    filled = filled.T
    Z = np.ma.masked_where(~filled, Z)
    f = ax.pcolormesh(X, Y, Z, cmap=mpl.colormaps[cmap])
    addColorbar(ax, f)
    return f


def plotHist(ax, edges, vals):
    def addColorbar(ax, vals):
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cbar = plt.colorbar(vals, cax=cax)
        cax.get_yaxis().set_offset_position("left")
        ax.cax = cax

    X, Y = np.meshgrid(*edges)
    Z = vals
    Z = np.ma.masked_where(~filled, Z)
    f = ax.pcolormesh(X, Y, Z)
    addColorbar(ax, f)
    return f
