import matplotlib.pyplot as plt
import numpy as np

from src.constants import DEFAULT_SEED
from src.models.knn import run_experiment


def _protocol_b(
    k_values,
    v,
    n_dims,
    number_of_runs,
    number_of_real_points,
    number_of_train_points_vector,
    number_of_test_points,
):
    error = np.zeros(
        (len(number_of_train_points_vector), len(k_values), number_of_runs)
    )
    np.random.seed(DEFAULT_SEED)
    seeds = np.random.randint(
        low=len(number_of_train_points_vector) * len(k_values) * number_of_runs,
        size=(len(number_of_train_points_vector), len(k_values), number_of_runs),
    )
    for i, number_of_train_points in enumerate(number_of_train_points_vector):
        for j, k_value in enumerate(k_values):
            for k in range(number_of_runs):
                error[i, j, k] = run_experiment(
                    k_value,
                    v,
                    n_dims,
                    number_of_real_points,
                    number_of_train_points,
                    number_of_test_points,
                    seed=seeds[i, j, k],
                )
    return error


def all_parts(
    k_values,
    v,
    n_dims,
    number_of_runs,
    number_of_real_points,
    number_of_train_points_vector,
    number_of_test_points,
    figure_title,
    figure_path,
):
    error = _protocol_b(
        k_values,
        v,
        n_dims,
        number_of_runs,
        number_of_real_points,
        number_of_train_points_vector,
        number_of_test_points,
    )
    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(10)
    fig.set_figheight(10)
    plt.plot(
        number_of_train_points_vector,
        np.mean(k_values[np.argmin(error, axis=1)], axis=1),
    )
    plt.title(figure_title)
    plt.xlabel("Number of Training Points (m)")
    plt.ylabel("Optimal k-Value")
    plt.xticks(number_of_train_points_vector)
    plt.savefig(figure_path)
