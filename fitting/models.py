import sys

import gpytorch
import torch


class GaussianMean(gpytorch.means.Mean):
    def __init__(self, prior=None, init_mean=0.0, init_sigma=1.0, init_scale=1.0):
        super().__init__()
        self.register_parameter(
            name="mean",
            parameter=torch.nn.Parameter(torch.tensor(init_mean, dtype=torch.float64)),
        )
        self.register_parameter(
            name="sigma",
            parameter=torch.nn.Parameter(torch.tensor(init_sigma, dtype=torch.float64)),
        )
        self.register_parameter(
            name="scale",
            parameter=torch.nn.Parameter(torch.tensor(init_scale, dtype=torch.float64)),
        )
        if prior is not None:
            self.register_prior("contant_prior", prior, "mean")
            self.register_prior("constant_prior", prior, "sigma")
            self.register_prior("constant_prior", prior, "scale")

    def forward(self, input):
        inner = (input - self.mean) ** 2 @ (1 / self.sigma**2)
        e = torch.exp(-inner)
        ret = self.scale * e
        return ret


class MatrixRBF(gpytorch.kernels.Kernel):
    is_stationary = True

    def __init__(self, kmatrix_prior=None, kmatrix_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_kmatrix",
            parameter=torch.nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]])),
        )

        # set the parameter constraint to be positive, when nothing is specified
        if kmatrix_constraint is None:
            kmatrix_constraint = gpytorch.constraints.Interval(-30, 30)

        # register the constraint

        self.register_constraint("raw_kmatrix", kmatrix_constraint)

        # set the parameter prior, see
        # https://docs.gpytorch.ai/en/latest/module.html#gpytorch.Module.register_prior
        if kmatrix_prior is not None:
            self.register_prior(
                "kmatrix_prior",
                kmatrix_prior,
                lambda m: m.kmatrix,
                lambda m, v: m._set_kmatrix(v),
            )

        # self.kmatrix = torch.tensor([[1.0, -1.0], [-1.0, 1.0]])

    # now set up the 'actual' paramter
    @property
    def kmatrix(self):
        # when accessing the parameter, apply the constraint transform
        return self.raw_kmatrix_constraint.transform(self.raw_kmatrix)

    @kmatrix.setter
    def kmatrix(self, value):
        return self._set_kmatrix(value)

    def _set_kmatrix(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_kmatrix)
        # when setting the paramater, transform the actual value to a raw one by applying the inverse transform
        self.initialize(
            raw_kmatrix=self.raw_kmatrix_constraint.inverse_transform(value)
        )

    # this is the kernel function
    def forward(self, x1, x2, diag=False, **params):
        diff = torch.unsqueeze(x1, dim=1) - x2
        transformed = diff @ self.kmatrix
        mat = torch.einsum("ijk,ijk->ij", transformed, diff)
        covar = torch.exp(-mat)
        # print(self.kmatrix)
        if diag:
            return covar.diagonal()
        else:
            return covar


class ExactProjGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = mean or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.register_parameter
        # self.proj_mat = torch.nn.Parameter(torch.tensor([[1,1],[1,0]], dtype=torch.float64))
        self.rot = torch.nn.Parameter(torch.tensor(0.78, dtype=torch.float64))

    def forward(self, x):
        rot_mat = torch.tensor(
            [
                [torch.cos(self.rot), -torch.sin(self.rot)],
                [torch.sin(self.rot), torch.cos(self.rot)],
            ]
        )
        x = x @ rot_mat  # n x d * d x k --> n x k
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, mean=None):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = mean or gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactAnyKernelModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood, kernel=None, mean=None):
        super().__init__(train_x, train_y, likelihood)
        # self.mean_module = gpytorch.means.ConstantMean()
        self.mean_module = mean or gpytorch.means.ConstantMean()
        if kernel is None:
            kernel = gpytorch.kernels.ScaleKernel(
                gpytorch.kernels.RBFKernel(ard_num_dims=2)
            )
        self.covar_module = kernel

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)

        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class ExactPeakedGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )

        self.covar_peak_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(ard_num_dims=2)
        )
        self.peak = torch.nn.Parameter(torch.tensor([0.8, 0.2], dtype=torch.float64))

    def forward(self, x):
        subbed = x - self.peak
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x) * self.covar_peak_module(subbed)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


class PeakedRBF(gpytorch.kernels.RBFKernel):
    is_stationary = False

    def __init__(self, peak_prior=None, peak_constraint=None, **kwargs):
        super().__init__(**kwargs)
        self.register_parameter(
            name="raw_peak",
            parameter=torch.nn.Parameter(torch.tensor([0.1, 0.1], requires_grad=True)),
        )
        if peak_constraint is None:
            peak_constraint = gpytorch.constraints.Positive()
        self.register_constraint("raw_peak", peak_constraint)
        if peak_prior is not None:
            self.register_prior(
                "peak_prior",
                peak_prior,
                lambda m: m.peak,
                lambda m, v: m._set_peak(v),
            )

    @property
    def peak(self):
        return self.raw_peak_constraint.transform(self.raw_peak)

    @peak.setter
    def peak(self, value):
        return self._set_peak(value)

    def _set_peak(self, value):
        if not torch.is_tensor(value):
            value = torch.as_tensor(value).to(self.raw_peak)
        self.initialize(raw_peak=self.raw_peak_constraint.inverse_transform(value))

    def forward(self, x1, x2, **params):
        return super().forward(x1 - self.peak, x2 - self.peak, **params)
