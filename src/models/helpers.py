import numpy as np


def least_square_solution(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return np.linalg.solve(x.T @ x, x.T @ y)


def phi_polynomial(x: np.ndarray, k: int):
    return np.concatenate([x**i for i in range(k - 1, -1, -1)], axis=1)


def phi_sin(x: np.ndarray, k: int):
    return np.concatenate([np.sin(i * np.pi * x) for i in range(k, 0, -1)], axis=1)


def mean_squared_error(y_actual: np.ndarray, y_predicted: np.ndarray):
    return np.mean((y_actual - y_predicted) ** 2)


def train_test_split(x, y, split):
    n = x.shape[0]
    indices = np.random.choice(range(n), int(split * n))
    mask = np.ones(n, dtype=bool)
    mask[indices] = False
    return x[indices, :], y[indices], x[mask, :], y[mask]
