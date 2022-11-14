from abc import ABC, abstractmethod

import jax.numpy as jnp
import numpy as np
from jax import vmap

from src.models.helpers import least_square_solution


class RegressionModel(ABC):
    def __init__(self):
        self.coefficients = None
        self.x_train = None

    @abstractmethod
    def fit(self, x_train, y_train):
        raise NotImplementedError("Please Implement this method")

    @abstractmethod
    def predict(self, x_test):
        raise NotImplementedError("Please Implement this method")

    def fit_predict(self, x_train, y_train, x_test):
        self.fit(x_train, y_train)
        return self.predict(x_test)

    def reset(self):
        self.coefficients = None
        self.x_train = None

    @staticmethod
    def add_ones(X):
        return np.c_[np.ones(X.shape[0]), X]


class NaiveLinearRegression(RegressionModel):
    def __init__(self):
        self.coefficients = None
        super().__init__()

    def fit(self, x_train, y_train):
        self.coefficients = np.mean(y_train)

    def predict(self, x_test):
        return np.ones(x_test.shape[0]) * self.coefficients


class SingleLinearRegression(RegressionModel):
    def __init__(self, idx):
        self.coefficients = None
        self.idx = idx
        super().__init__()

    def fit(self, x_train, y_train):
        self.coefficients = least_square_solution(
            self.add_ones(x_train[:, self.idx]), y_train
        )

    def predict(self, x_test):
        return self.add_ones(x_test[:, self.idx]) @ self.coefficients


class MultipleLinearRegression(RegressionModel):
    def __init__(self):
        self.coefficients = None
        super().__init__()
        self.x_train = None

    def fit(self, x_train, y_train):
        self.coefficients = least_square_solution(self.add_ones(x_train), y_train)

    def predict(self, x_test):
        return self.add_ones(x_test) @ self.coefficients


class GaussianKernelRidgeRegression(RegressionModel):
    def __init__(self, gamma: float, sigma: float):
        self.coefficients: np.ndarray = None  # (N_train ,1)
        self.gamma: float = gamma
        self.sigma: float = sigma
        self.x_train: np.ndarray = None
        super().__init__()

    @staticmethod
    def k(x, y, sigma):
        return jnp.exp(-(x - y).T@(x - y) / (2*sigma**2))

    def gram(self, x1, x2):
        """

        :param x1: (N_train, D)
        :param x2: (N_train, D)
        :return: (N_train, N_train)
        """
        return vmap(lambda x: vmap(lambda y: self.k(x, y, self.sigma))(x2))(x1)

    def fit(self, x_train, y_train):
        n = x_train.shape[0]
        self.coefficients, residual, _, _ = jnp.linalg.lstsq(
            self.gram(x_train, x_train) + self.gamma * n * jnp.identity(n),
            y_train,
            rcond=None,
        )
        self.x_train = x_train

    def predict(self, x_test):
        return jnp.dot(self.gram(x_test, self.x_train), self.coefficients)
