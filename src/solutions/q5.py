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


def find_optimal_parameters(x, y, number_of_folds, log_2_gammas, log_2_sigmas):
    mses = np.zeros((len(log_2_sigmas), len(log_2_gammas), number_of_folds))
    for i, log_2_sigma in enumerate(log_2_sigmas):
        for j, log_2_gamma in enumerate(log_2_gammas):
            folds = make_folds(x, y, number_of_folds)
            kernel = GaussianKernel(sigma=2.0**log_2_sigma)
            model = KernelRidgeRegression(kernel, 2.0**log_2_gamma)
            for k, fold in enumerate(folds):
                model.fit(fold.x_train, fold.y_train)
                mses[i, j, k] = mean_squared_error(
                    model.predict(fold.x_test), fold.y_test
                )
    average_mses = np.mean(mses, axis=-1)
    best_sigma_idx, best_gamma_idx = np.unravel_index(
        average_mses.argmin(), average_mses.shape
    )
    return (
        log_2_gammas[best_gamma_idx].item(),
        log_2_sigmas[best_sigma_idx].item(),
        mses,
    )


def _compute_kernel_ridge_regression_performance(
    x,
    y,
    train_percentage: float,
    number_of_runs: int,
    model_name: str,
    number_of_folds,
    log_2_gammas,
    log_2_sigmas,
):
    train_mses = []
    test_mses = []
    mses = []
    best_log_2_gammas = []
    best_log_2_sigmas = []
    for _ in range(number_of_runs):
        train_test_data = split_train_test_data(x, y, train_percentage)
        log_2_gamma, log_2_sigma, mse = find_optimal_parameters(
            x=train_test_data.x_train,
            y=train_test_data.y_train,
            number_of_folds=number_of_folds,
            log_2_gammas=log_2_gammas,
            log_2_sigmas=log_2_sigmas,
        )
        kernel = GaussianKernel(sigma=2.0**log_2_sigma)
        model = KernelRidgeRegression(kernel, gamma=2.0**log_2_gamma)
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
        best_log_2_gammas.append(log_2_gamma)
        best_log_2_sigmas.append(log_2_sigma)
        mses.append(mse)
    df_performance = pd.DataFrame(
        data=[
            [
                model_name,
                np.round(np.mean(np.array(train_mses)), 2),
                np.round(np.std(np.array(train_mses)), 2),
                np.round(np.mean(np.array(test_mses)), 2),
                np.round(np.std(np.array(test_mses)), 2),
            ]
        ],
        columns=PERFORMANCE_COLUMN_NAMES,
    )
    df_params = pd.DataFrame(
        data=np.array(
            [
                np.arange(number_of_runs).astype(int),
                best_log_2_gammas,
                best_log_2_sigmas,
            ]
        ).T,
        columns=["Trial", "Log_2 Sigma", "Log_2 Gamma"],
    )
    df_params["Trial"] = df_params["Trial"].astype(int)
    return df_performance.set_index(PERFORMANCE_COLUMN_NAMES[0]), mses, df_params


def abc(
    df,
    feature_columns,
    target_column,
    train_percentage: float,
    number_of_folds: int,
    log_2_gammas,
    log_2_sigmas,
    optimal_params_csv_path,
    performance_csv_path,
    mse_figure_path,
):
    number_of_runs = 1
    (
        df_ridge_regression_performance,
        mses,
        df_params,
    ) = _compute_kernel_ridge_regression_performance(
        df[feature_columns].values[:],
        df[[target_column]].values[:],
        train_percentage,
        number_of_runs,
        _MODEL_NAME,
        number_of_folds,
        log_2_gammas,
        log_2_sigmas,
    )

    df_ridge_regression_performance.to_csv(performance_csv_path)
    df_params.to_csv(optimal_params_csv_path)
    average_mses = np.mean(mses[0], axis=-1)
    plt.figure()
    plt.imshow(np.log(average_mses))
    plt.colorbar()
    plt.xticks(
        np.arange(len(log_2_gammas)),
        labels=log_2_gammas.reshape(-1),
    )
    plt.xlabel("log_2(gamma)")
    plt.yticks(
        np.arange(len(log_2_sigmas)),
        labels=log_2_sigmas.reshape(-1),
    )
    plt.ylabel("log_2(sigma)")
    plt.title(f"log(mean {number_of_folds}-fold MSE)")
    plt.savefig(mse_figure_path)


def d(
    df,
    feature_columns,
    target_column,
    train_percentage: float,
    number_of_runs: int,
    number_of_folds,
    log_2_gammas,
    log_2_sigmas,
    q4_performance_csv_path,
    performance_csv_path,
):
    (
        df_ridge_regression_performance,
        _,
        _,
    ) = _compute_kernel_ridge_regression_performance(
        df[feature_columns].values[:],
        df[[target_column]].values[:],
        train_percentage,
        number_of_runs,
        _MODEL_NAME,
        number_of_folds,
        log_2_gammas,
        log_2_sigmas,
    )
    df_performance_q4 = pd.read_csv(q4_performance_csv_path)
    df_performance = pd.DataFrame(
        list(df_performance_q4.values[:])
        + list(df_ridge_regression_performance.reset_index().values[:]),
        columns=PERFORMANCE_COLUMN_NAMES,
    ).round(2)
    df_performance.set_index(PERFORMANCE_COLUMN_NAMES[0]).to_csv(performance_csv_path)
