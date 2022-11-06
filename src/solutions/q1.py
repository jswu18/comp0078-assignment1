import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.helpers import least_square_solution, mean_squared_error, phi_polynomial


def _fit(x, y, ks):
    weights = {}
    for k in ks:
        weights[k] = least_square_solution(phi_polynomial(x, k), y)
    return weights


def a(x, y, ks, figure_title, figure_path):
    weights = _fit(x, y, ks)
    x_plot = np.linspace(1, 4, 100).reshape(-1, 1)
    plt.figure()
    for k in ks:
        mse = np.round(
            mean_squared_error(y, np.dot(phi_polynomial(x, k), weights[k])), 2
        )
        plt.scatter(x, y)
        plt.plot(
            x_plot, np.dot(phi_polynomial(x_plot, k), weights[k]), label=f"{k=}, {mse=}"
        )
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(figure_title)
    plt.legend()
    plt.savefig(figure_path)


def b(x, y, ks, table_path):
    weights = _fit(x, y, ks)
    w = np.zeros((len(ks), np.max(ks)))
    for i, k in enumerate(ks):
        w[i, :k] = weights[k].reshape(-1)
    df = pd.DataFrame(w.T, columns=[f"{k=}" for k in ks])
    df.index = ["1"] + [f"x^{i}" for i in range(1, np.max(ks))]
    df.index.name = "Basis"
    df.to_csv(table_path)


def c(x, y, ks, table_path):
    weights = _fit(x, y, ks)
    mse = np.zeros((len(weights), 1))
    for i, k in enumerate(ks):
        mse[i] = np.round(
            mean_squared_error(y, np.dot(phi_polynomial(x, k), weights[k])), 2
        )
    df = pd.DataFrame(mse.T, columns=[f"{k=}" for k in ks])
    df.index = ["MSE"]
    df.index.name = "Metric"
    df.to_csv(table_path)
