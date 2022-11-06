import matplotlib.pyplot as plt
import numpy as np

from src.constants import DEFAULT_SEED
from src.models.knn import run_experiment


def _protocol_a(
    k_values,
    v,
    n_dims,
    number_of_runs,
    number_of_real_points,
    number_of_train_points,
    number_of_test_points,
):
    performance = np.zeros((len(k_values), number_of_runs))
    np.random.seed(DEFAULT_SEED)
    seeds = np.random.randint(
        low=len(k_values) * number_of_runs, size=(len(k_values), number_of_runs)
    )
    for i, k_value in enumerate(k_values):
        for j in range(number_of_runs):
            performance[i, j] = run_experiment(
                k_value,
                v,
                n_dims,
                number_of_real_points,
                number_of_train_points,
                number_of_test_points,
                seed=seeds[i, j],
            )
    return performance


def all_parts(
    k_values,
    v,
    n_dims,
    number_of_runs,
    number_of_real_points,
    number_of_train_points,
    number_of_test_points,
    figure_title,
    figure_path,
):
    performance = _protocol_a(
        k_values,
        v,
        n_dims,
        number_of_runs,
        number_of_real_points,
        number_of_train_points,
        number_of_test_points,
    )
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(15)
    fig.set_figheight(5)
    plt.plot(k_values, np.mean(performance, axis=1))
    plt.title(figure_title)
    plt.xticks(k_values)
    plt.xlabel("k Value")
    plt.ylabel("Mean Test Set Error")
    plt.savefig(figure_path)
