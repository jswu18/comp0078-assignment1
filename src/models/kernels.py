from abc import ABC, abstractmethod

import jax.numpy as jnp
from jax import vmap


class Kernel(ABC):
    @abstractmethod
    def k(self, x, y):
        pass

    def compute_gram(self, x, y):
        return vmap(
            lambda x_i: vmap(
                lambda y_i: self.k(x_i, y_i)
            )(y)
        )(x)


class GaussianKernel(Kernel):
    def __init__(self, sigma):
        self.sigma = sigma

    def k(self, x, y):
        return jnp.exp(-((x - y).T @ (x - y)) / (2 * (self.sigma**2)))


class LinearKernel(Kernel):
    def __init__(self, c):
        self.c = c

    def k(self, x, y):
        return self.c + x@y

