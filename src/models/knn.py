from typing import Tuple

import numpy as np

from src.constants import DEFAULT_SEED


def get_p_h(
    number_of_points: int, n_dims: int, seed: int = DEFAULT_SEED
) -> Tuple[np.ndarray, np.ndarray]:
    """
    generate data for a voted-center hypothesis
    """
    np.random.seed(seed)
    x = np.random.uniform(low=0, high=1, size=(number_of_points, n_dims))
    y = np.random.choice(a=[0, 1], size=number_of_points)
    return x, y


def _all_distances(a, b):
    """
    a is (N, D)
    b is (M, D)
    returns all euclidean norm distances between a and b (N, M)

    """

    return np.sqrt(((a.T[:, None] - b.T[:, :, None]) ** 2).sum(0)).T


def knn_prediction(
    train_data: Tuple[np.ndarray, np.ndarray], test_points: np.ndarray, k: int
) -> np.ndarray:
    """
    Make knn prediction on test_points using train_data

    train_data: Tuple of x, y for training
    test_points: numpy array of test points
    k: an integer

    return label for each test point
    """
    train_x, train_y = train_data
    euclidean_distances = _all_distances(test_points, train_x)
    sorted_euclidean_distance_idxs = np.argsort(euclidean_distances, axis=1)
    k_closest_idxs = sorted_euclidean_distance_idxs[:, :k]

    vote_percentages = np.sum(train_y[k_closest_idxs], axis=1) / k

    # find_undefined cases and choose randomly
    idx_undefined = np.where(vote_percentages == 0.5)[0]
    vote_percentages[idx_undefined] = np.random.choice(
        a=[0, 1], size=len(idx_undefined)
    )

    # choose label
    vote = np.round(vote_percentages)
    return vote


def _noisy_knn_prediction(
    train_data: Tuple[np.ndarray, np.ndarray], test_points: np.ndarray, k: int
) -> np.ndarray:
    """
    do knn but randomly choose label 20% of the time
    """
    # flip coin
    coin_flip = np.random.choice(
        a=[0, 1], size=len(test_points), p=[0.2, 0.8]
    )  # 0 is tails, 1 is head

    # use knn to generate label
    labels = knn_prediction(train_data, test_points, k)

    # generate random y
    noise_labels = np.random.choice(a=[0, 1], size=len(test_points))

    # replace actual labels with noise where coin is tails
    idx_tails = np.where(coin_flip == 0)
    labels[idx_tails] = noise_labels[idx_tails]
    return labels


def generate_noisy_data(
    p_h: Tuple[np.ndarray, np.ndarray],
    number_samples: int,
    v: int,
    n_dims: int,
    seed: int = DEFAULT_SEED,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    p_h: data for a voted-center hypothesis
    number_samples: number of noisy data points to generate
    v: number of neighbours to use during data generation

    return x and noisy labels
    """
    # generate x uniformly
    np.random.seed(seed)
    x = np.random.uniform(low=0, high=1, size=(number_samples, n_dims))
    noisy_labels = _noisy_knn_prediction(train_data=p_h, test_points=x, k=v)
    return x, noisy_labels


def run_experiment(
    k_value,
    v,
    n_dims,
    number_of_real_points,
    number_of_train_points,
    number_of_test_points,
    seed,
):
    p_h = get_p_h(number_of_real_points, n_dims, seed)
    data_train = generate_noisy_data(
        p_h, number_samples=number_of_train_points, v=v, n_dims=n_dims
    )
    data_test = generate_noisy_data(
        p_h, number_samples=number_of_test_points, v=v, n_dims=n_dims
    )

    x_test, y_test = data_test
    y_test_predicted = knn_prediction(data_train, x_test, k_value)
    return np.sum(y_test_predicted != y_test) / number_of_test_points
