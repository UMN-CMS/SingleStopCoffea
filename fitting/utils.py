import linear_operator
import torch


def fixMVN(mvn):
    X = linear_operator.utils.cholesky.psd_safe_cholesky(mvn.covariance_matrix)
    fixed_cv = X @ X.T
    mu = mvn.mean
    fixed = type(mvn)(mu, fixed_cv)
    # assert(torch.allclose(fixed_cv, mvn.covariance_matrix))
    return fixed


def affineTransformMVN(mvn, slope, intercept):
    cm = mvn.covariance_matrix
    mu = mvn.mean
    new_mu = slope * mu + intercept
    d = slope * torch.eye(cm.shape[0])
    new_cm = d @ cm @ d.T
    ret = type(mvn)(new_mu, new_cm)
    return ret


def pointsToGrid(points_x, points_y, edges, set_unfilled=None):
    filled = torch.histogramdd(
        points_x, bins=edges, weight=torch.full_like(points_y, True)
    )
    ret = torch.histogramdd(points_x, bins=edges, weight=points_y)
    return ret, filled.hist.bool()
