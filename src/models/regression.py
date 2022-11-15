from abc import ABC, abstractmethod

import numpy as np

from src.models.helpers import least_square_solution
from src.models.kernels import Kernel


class Regression(ABC):
    def __init__(self):
        self.w: np.ndarray = None

    @staticmethod
    def phi(x):
        pass

    @abstractmethod
    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        """
        x (n, d)
        y (n, 1)
        return (d, 1)
        """
        pass

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x (n, d)
        """
        return x @ self.w


class NaiveRegression(Regression):
    def __init__(self):
        super().__init__()

    @staticmethod
    def phi(x):
        return np.ones((x.shape[0], 1))

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.w = least_square_solution(self.phi(x), y)

    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        x (n, d)
        """
        return self.phi(x) @ self.w


class LinearRegression(NaiveRegression):
    def __init__(self):
        super().__init__()

    @staticmethod
    def phi(x):
        return np.concatenate((x, np.ones((x.shape[0], 1))), axis=1)


class KernelRidgeRegression(Regression):
    def __init__(self, kernel: Kernel, gamma: float):
        self.kernel = kernel
        self.gamma = gamma
        self.x_train: np.ndarray = None
        super().__init__()

    @staticmethod
    def compute_k_regularised(x, kernel, gamma):
        n, _ = x.shape
        return kernel.compute_gram(x, x) + gamma * n * np.eye(n)

    def fit(self, x: np.ndarray, y: np.ndarray) -> None:
        self.x_train = x
        self.w = (
            np.linalg.inv(self.compute_k_regularised(x, self.kernel, self.gamma)) @ y
        )

    def predict(self, x):
        return self.kernel.compute_gram(x, self.x_train) @ self.w
