import matplotlib.pyplot as plt
import numpy as np

from src.constants import DEFAULT_SEED
from src.models.helpers import least_square_solution, mean_squared_error, phi_polynomial


def _g(x: np.ndarray) -> np.ndarray:
    return np.sin(2 * np.pi * x) ** 2


def _g_sigma(x: np.ndarray, sigma: float) -> np.ndarray:
    e = np.random.normal(0, sigma, size=x.shape)
    return _g(x) + e


def _g_noisy_samples(number_of_sample_points, sigma):
    x_samples = np.random.uniform(low=0, high=1, size=(number_of_sample_points, 1))
    y_samples_noisy = _g_sigma(x_samples, sigma).reshape(-1, 1)
    return x_samples, y_samples_noisy


def _calculate_mse(
    number_of_sample_points, sigma, number_test_points, number_of_trials, phi, bases
):
    test_mse = np.zeros((number_of_trials, len(bases)))

    np.random.seed(DEFAULT_SEED)
    seeds = np.random.randint(low=number_of_trials, size=(number_of_trials,))
    for i in range(number_of_trials):
        np.random.seed(seeds[i])
        x_samples, y_samples_noisy = _g_noisy_samples(
            number_of_sample_points,
            sigma,
        )
        weights_dict = {}
        for base in bases:
            weights_dict[base] = least_square_solution(
                phi(x_samples, base), y_samples_noisy
            )
        for j, base in enumerate(weights_dict.keys()):
            x_test = np.random.uniform(low=0, high=1, size=(number_test_points, 1))
            y_test_actual = _g(x_test)
            y_test_prediction = np.dot(phi(x_test, base), weights_dict[base])
            test_mse[i, j] = mean_squared_error(y_test_actual, y_test_prediction)
    return test_mse


def a_i(number_of_sample_points, sigma, figure_title, figure_path):
    x_samples, y_samples_noisy = _g_noisy_samples(number_of_sample_points, sigma)
    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    y_plot = _g(x_plot)

    plt.figure()
    plt.plot(x_plot, y_plot)
    plt.scatter(x_samples, y_samples_noisy)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title(figure_title)
    plt.savefig(figure_path)


def a_ii(
    number_of_sample_points, sigma, ks, figure_title, figure_path, phi=phi_polynomial
):
    x_samples, y_samples_noisy = _g_noisy_samples(number_of_sample_points, sigma)
    weights = {}
    for k in ks:
        weights[k] = least_square_solution(phi(x_samples, k), y_samples_noisy)

    x_plot = np.linspace(0, 1, 100).reshape(-1, 1)
    plt.figure()
    for k in ks:
        mse = np.round(
            mean_squared_error(y_samples_noisy, np.dot(phi(x_samples, k), weights[k])),
            2,
        )
        plt.scatter(x_samples, y_samples_noisy)
        plt.plot(x_plot, np.dot(phi(x_plot, k), weights[k]), label=f"{k=}, {mse=}")
    plt.title(figure_title)
    plt.xlabel("x")
    plt.ylabel("y")
    plt.ylim([-2.5, 2.5])
    plt.legend()
    plt.savefig(figure_path)


def b(number_of_sample_points, sigma, ks, figure_title, figure_path, phi):
    weights = {}
    x_samples, y_samples_noisy = _g_noisy_samples(number_of_sample_points, sigma)
    for k in ks:
        weights[k] = least_square_solution(phi(x_samples, k), y_samples_noisy)

    mse_train = np.array(
        [
            mean_squared_error(y_samples_noisy, np.dot(phi(x_samples, k), weights[k]))
            for k in ks
        ]
    )

    plt.figure()
    plt.plot(ks, np.log(mse_train))
    plt.title(figure_title)
    plt.xlabel("Number of Bases (k)")
    plt.xticks(ks)
    plt.ylabel("ln(MSE)")
    plt.savefig(figure_path)


def c_and_d(
    number_of_sample_points,
    sigma,
    number_test_points,
    number_of_trials,
    ks,
    figure_title,
    figure_path,
    phi,
):
    test_mse = _calculate_mse(
        number_of_sample_points,
        sigma,
        number_test_points,
        number_of_trials,
        phi=phi,
        bases=ks,
    )
    plt.figure()
    plt.plot(ks, np.log(np.mean(test_mse, axis=0)))
    plt.title(f"{figure_title} ({number_of_trials=})")
    plt.xlabel("Number of Bases (k)")
    plt.xticks(ks)
    plt.ylabel("ln(avg(MSE))")
    plt.savefig(figure_path)
