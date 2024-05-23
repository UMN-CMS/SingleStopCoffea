from dataclasses import asdict, dataclass
from typing import Tuple

import torch


@dataclass
class RectWindow:
    x_range: Tuple[float, float]
    y_range: Tuple[float, float]

    def __call__(self, X, Y):
        a = (X > self.x_range[0]) & (X < self.x_range[1])
        b = (Y > self.y_range[0]) & (Y < self.y_range[1])
        return a, b

    def toString(self):
        return f"E__{self.x_range[0]}_{self.x_range[1]}_{self.y_range[0]}_{self.y_range[1]}".replace(
            ".", "p"
        )

    def toDict(self):
        return asdict(self)


@dataclass
class EllipseWindow:
    center: Tuple[float, float]
    axes: Tuple[float, float]

    def __call__(self, X, Y):
        axes = torch.tensor(self.axes)
        center = torch.tensor(self.center)
        stacked = torch.stack((X, Y), axis=-1)
        rel = ((stacked - center) ** 2) / axes**2
        mask = (torch.select(rel, -1, 0) + torch.select(rel, -1, 1)) <= 1.0
        return mask, mask

    def toString(self):
        return f"E__{self.center[0]}_{self.center[1]}_{self.axes[0]}_{self.axes[1]}".replace(
            ".", "p"
        )

    def toDict(self):
        return asdict(self)
