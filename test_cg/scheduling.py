# Todo：The subsequent organization will be incorporated into the config file; it will not exist as a separate file.
import torch
from torch import Tensor

class BaseScheduler:
    def alpha(self, t) -> Tensor:
        ...
    def sigma(self, t) -> Tensor:
        ...

    def dalpha(self, t) -> Tensor:
        ...
    def dsigma(self, t) -> Tensor:
        ...

    def dalpha_over_alpha(self, t) -> Tensor:
        return self.dalpha(t) / self.alpha(t)

    def dsigma_mul_sigma(self, t) -> Tensor:
        return self.dsigma(t)*self.sigma(t)

    def drift_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dalpha/(alpha + 1e-6)

    def diffuse_coefficient(self, t):
        alpha, sigma = self.alpha(t), self.sigma(t)
        dalpha, dsigma = self.dalpha(t), self.dsigma(t)
        return dsigma*sigma - dalpha/(alpha + 1e-6)*sigma**2

    def w(self, t):
        return self.sigma(t)

import math


class LinearScheduler(BaseScheduler):
    def alpha(self, t) -> Tensor:
        return (t).view(-1, 1, 1, 1)
    def sigma(self, t) -> Tensor:
        return (1-t).view(-1, 1, 1, 1)
    def dalpha(self, t) -> Tensor:
        return torch.full_like(t, 1.0).view(-1, 1, 1, 1)
    def dsigma(self, t) -> Tensor:
        return torch.full_like(t, -1.0).view(-1, 1, 1, 1)

# SoTA for ImageNet!
class GVPScheduler(BaseScheduler):
    def alpha(self, t) -> Tensor:
        return torch.cos(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def sigma(self, t) -> Tensor:
        return torch.sin(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def dalpha(self, t) -> Tensor:
        return -torch.sin(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def dsigma(self, t) -> Tensor:
        return torch.cos(t * (math.pi / 2)).view(-1, 1, 1, 1)
    def w(self, t):
        return torch.sin(t)**2

class ConstScheduler(BaseScheduler):
    def w(self, t):
        return torch.ones(1, 1, 1, 1).to(t.device, t.dtype)




