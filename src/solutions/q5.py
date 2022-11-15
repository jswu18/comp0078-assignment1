import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.models.helpers import (
    TrainTestData,
    make_folds,
    mean_squared_error,
    split_train_test_data,
)
from src.models.kernels import GaussianKernel
from src.models.regression import KernelRidgeRegression
from src.solutions.q4 import PERFORMANCE_COLUMN_NAMES

_MODEL_NAME = "Kernel Ridge Regression"


def _convert_to_scientific_notation(x: float) -> str:
    """
    Convert value to string in scientific notation
    :param x: value to convert
    :return: string of x in scientific notation
    """
    return "{:.1e}".format(float(x))


def _compute_model_performance(train_test_data: TrainTestData, gamma, sigma):
    kernel = GaussianKernel(sigma=sigma)
    model = KernelRidgeRegression(kernel, gamma)
    model.fit(train_test_data.x_train, train_test_data.y_train)
    return mean_squared_error(
        model.predict(train_test_data.x_test), train_test_data.y_test
    )


def find_optimal_parameters(x, y, number_of_folds, gammas, sigmas):
    folds = make_folds(x, y, number_of_folds)
    mses = np.zeros((len(sigmas), len(gammas), number_of_folds))
    for i, sigma in enumerate(sigmas):
        for j, gamma in enumerate(gammas):
            for k, fold in enumerate(folds):
                kernel = GaussianKernel(sigma=sigma)
                model = KernelRidgeRegression(kernel, gamma)
                model.fit(fold.x_train, fold.y_train)
                mses[i, j, k] = mean_squared_error(
                    model.predict(fold.x_test), fold.y_test
                )
    average_mses = np.mean(mses, axis=-1)
    best_sigma_idx, best_gamma_idx = np.unravel_index(
        average_mses.argmin(), average_mses.shape
    )
    return gammas[best_gamma_idx].item(), sigmas[best_sigma_idx].item(), mses


def _compute_kernel_ridge_regression_performance(
    x,
    y,
    train_percentage: float,
    number_of_runs: int,
    model_name: str,
    number_of_folds,
    gammas,
    sigmas,
):
    train_mses = []
    test_mses = []
    mses = []
    for _ in range(number_of_runs):
        train_test_data = split_train_test_data(x, y, train_percentage)
        gamma, sigma, mse = find_optimal_parameters(
            x=train_test_data.x_train,
            y=train_test_data.y_train,
            number_of_folds=number_of_folds,
            gammas=gammas,
            sigmas=sigmas,
        )

        kernel = GaussianKernel(sigma=sigma)
        model = KernelRidgeRegression(kernel, gamma)
        model.fit(train_test_data.x_train, train_test_data.y_train)
        train_mses.append(
            mean_squared_error(
                model.predict(train_test_data.x_train), train_test_data.y_train
            )
        )
        test_mses.append(
            mean_squared_error(
                model.predict(train_test_data.x_test), train_test_data.y_test
            )
        )
        mses.append(mse)
    df_performance = pd.DataFrame(
        data=[
            [
                model_name,
                np.mean(train_mses),
                np.std(train_mses),
                np.mean(test_mses),
                np.std(test_mses),
            ]
        ],
        columns=PERFORMANCE_COLUMN_NAMES,
    )
    return df_performance.set_index(PERFORMANCE_COLUMN_NAMES[0]), mses


def abc(
    df,
    feature_columns,
    target_column,
    train_percentage: float,
    number_of_folds: int,
    gammas,
    sigmas,
    performance_csv_path,
    mse_figure_path,
):
    number_of_runs = 1
    (
        df_ridge_regression_performance,
        mses,
    ) = _compute_kernel_ridge_regression_performance(
        df[feature_columns].values[:],
        df[[target_column]].values[:],
        train_percentage,
        number_of_runs,
        _MODEL_NAME,
        number_of_folds,
        gammas,
        sigmas,
    )

    df_ridge_regression_performance.to_csv(performance_csv_path)

    fig, ax = plt.subplots()
    plt.imshow(np.log(np.mean(mses[0]), axis=-1))
    plt.xticks(
        np.arange(len(sigmas)),
        labels=[_convert_to_scientific_notation(x) for x in sigmas.reshape(-1)],
    )
    plt.yticks(
        np.arange(len(gammas)),
        labels=[_convert_to_scientific_notation(x) for x in gammas.reshape(-1)],
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fig.tight_layout()
    plt.savefig(mse_figure_path)


def d(
    df,
    feature_columns,
    target_column,
    train_percentage: float,
    number_of_runs: int,
    number_of_folds,
    gammas,
    sigmas,
    q4_performance_csv_path,
    performance_csv_path,
):
    df_ridge_regression_performance, _ = _compute_kernel_ridge_regression_performance(
        df[feature_columns].values[:],
        df[[target_column]].values[:],
        train_percentage,
        number_of_runs,
        _MODEL_NAME,
        number_of_folds,
        gammas,
        sigmas,
    )
    df_performance_q4 = pd.read_csv(q4_performance_csv_path)
    df_performance = pd.DataFrame(
        list(df_performance_q4.values[:])
        + list(df_ridge_regression_performance.reset_index().values[:]),
        columns=PERFORMANCE_COLUMN_NAMES,
    )
    df_performance.set_index(PERFORMANCE_COLUMN_NAMES[0]).to_csv(performance_csv_path)
