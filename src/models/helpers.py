from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class TrainTestData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray


def _build_train_test_data(x, y, indices):
    return TrainTestData(
        x_train=np.delete(x, indices, axis=0),
        y_train=np.delete(y, indices, axis=0),
        x_test=x[indices, :],
        y_test=y[indices, :],
    )


def split_train_test_data(
    x: np.ndarray, y: np.ndarray, train_percentage: float
) -> TrainTestData:
    n, _ = x.shape
    train_indices = np.random.choice(np.arange(n), int(train_percentage * n))
    return _build_train_test_data(x, y, indices=train_indices)


def make_folds(
    x: np.ndarray, y: np.ndarray, number_of_folds: int
) -> List[TrainTestData]:
    n, _ = x.shape
    indices = np.arange(n)
    np.random.shuffle(indices)
    split_indices = np.array_split(indices, number_of_folds)
    return [
        _build_train_test_data(x, y, indices=split_index)
        for split_index in split_indices
    ]


def least_square_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(x.T @ x, x.T @ y)


def phi_polynomial(x: np.ndarray, k: int):
    return np.concatenate([x**i for i in range(k - 1, -1, -1)], axis=1)


def phi_sin(x: np.ndarray, k: int):
    return np.concatenate([np.sin(i * np.pi * x) for i in range(k, 0, -1)], axis=1)


def mean_squared_error(y_actual: np.ndarray, y_predicted: np.ndarray):
    return np.mean((y_actual - y_predicted) ** 2)

