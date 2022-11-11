from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from jax import vmap

from mpl_toolkits import mplot3d

from src.models.helpers import mean_squared_error, train_test_split
from src.models.regression_models import (
    MultipleLinearRegression,
    NaiveLinearRegression,
    RegressionModel,
    SingleLinearRegression,
    GaussianKernelRidgeRegression
)

def _regression_report(models, x, y, iters):
    mse_results_test = np.zeros((iters, len(models)))
    mse_results_train = np.zeros((iters, len(models)))

    for i in range(iters):
        x_train, y_train, x_test, y_test = train_test_split(x, y, split=2 / 3)
        for j, model in enumerate(models):
            model.reset()
            mse_results_test[i, j] = mean_squared_error(
                model.fit_predict(x_train, y_train, x_test), y_test
            )
            
            mse_results_train[i, j] = mean_squared_error(
                model.predict(x_train), y_train
            )
    return np.mean(mse_results_test, axis=0), np.std(mse_results_test, axis=0), np.mean(mse_results_train, axis=0), np.std(mse_results_train, axis=0)


def _make_folds(x, k):
    indices = list(range(x.shape[0]))
    np.random.shuffle(indices)
    splits = np.array_split(indices, k)
    return splits


def _evaluate_mse_gkrr(gamma, sigma, x_train, y_train, x_test, y_test):
    model = GaussianKernelRidgeRegression(gamma, sigma)
    predictions = model.fit_predict(x_train=x_train, y_train=y_train, x_test=x_test)
    return mean_squared_error(predictions, y_test)


def _gkrr_model_selection(x, y, gammas, sigmas, k):
    mses = np.zeros((len(sigmas), len(gammas), k))
    folds = _make_folds(x, k)
    for idx, split in enumerate(folds):
        x_tr, y_tr, x_val, y_val = x[~split, :], y[~split], x[split, :], y[split]
        mses[:, :, idx] = vmap(
            lambda x_: vmap(
                lambda y_: _evaluate_mse_gkrr(y_, x_, x_tr, y_tr, x_val, y_val)
            )(gammas)
        )(sigmas)
    mses = np.mean(mses, axis=2)
    best_sigma_idx, best_gamma_idx = np.unravel_index(mses.argmin(), mses.shape)
    gamma = gammas[best_gamma_idx]
    sigma = sigmas[best_sigma_idx]
    return gamma, sigma, mses


def _regression_report_with_gkrr(models : List[RegressionModel], x, y, gammas, sigmas, iters):
    train_errors = np.zeros((iters, len(models) + 1))
    test_errors = train_errors.copy()
    for i in range(iters):
        x_train, y_train, x_test, y_test = train_test_split(x, y, split=2 / 3)
        for j, model in enumerate(models):
            model.reset()
            model.fit(x_train, y_train)
            train_errors[i, j] = mean_squared_error(model.predict(x_train), y_train)
            test_errors[i, j] = mean_squared_error(model.predict(x_test), y_test)
        gamma, sigma, _ = _gkrr_model_selection(
            x=x_train, y=y_train, gammas=gammas, sigmas=sigmas, k=5
        )
        model = GaussianKernelRidgeRegression(gamma, sigma)
        model.fit(x_train, y_train)
        train_errors[i, -1] = mean_squared_error(model.predict(x_train), y_train)
        test_errors[i, -1] = mean_squared_error(model.predict(x_test), y_test)
    return (
        np.mean(train_errors, axis=0),
        np.std(train_errors, axis=0),
        np.mean(test_errors, axis=0),
        np.std(test_errors, axis=0),
    )


def all_parts(
    x,
    y,
    data_columns,
    figure_title_train,
    figure_title_test,
    figure_path_train,
    figure_path_test,
    gkrr_param_path,
    contour_path,
    report_path):
    
    x_train, y_train, x_test, y_test = train_test_split(x, y, 2 / 3)

    models: List[RegressionModel] = []
    models += [SingleLinearRegression(idx) for idx in range(12)]
    models += [NaiveLinearRegression(), MultipleLinearRegression()]

    mses_test, stds_train, mses_train, mses_train = _regression_report(models, x_train, y_train, iters=20)

    axis_labels = data_columns[:12] + ["Naive", "MLR"]
    plt.figure(figsize=(15, 3))
    plt.bar(axis_labels, mses_test)
    plt.ylabel("MSE")
    plt.title(figure_title_test)
    plt.savefig(figure_path_test)

    axis_labels = data_columns[:12] + ["Naive", "MLR"]
    plt.figure(figsize=(15, 3))
    plt.bar(axis_labels, mses_train)
    plt.ylabel("MSE")
    plt.title(figure_title_train)
    plt.savefig(figure_path_train)

    gammas = np.array([2**x for x in list(range(-40,-27))])
    sigmas = 2**np.arange(7,13.5,0.5)

    gamma,sigma, mses = _gkrr_model_selection(x = x_train,y = y_train, gammas = gammas, sigmas = sigmas, k = 5)   # type: ignore
    GKRR_params = pd.DataFrame({'Optimal Gamma: ' : gamma, 'Optimal Sigma: ' : sigma, 'mse: ' : mses.min()}, index = [0])
    GKRR_params.to_csv(gkrr_param_path, index = False)
    fig = plt.figure()
    ax = plt.axes(projection='3d')
    ax.contour3D(gammas, sigmas, mses)  # type: ignore
    ax.set_xlabel('gamma')
    ax.set_ylabel('sigma')
    ax.set_zlabel('mse')  # type: ignore
    plt.savefig(contour_path)

    axis_labels += ['GKRR']
    train_mse, train_std, test_mse,test_std = _regression_report_with_gkrr(models, x,y,gammas,sigmas,20)
    df = pd.DataFrame(
        [train_mse,train_std,test_mse,test_std],
        columns =axis_labels,
        index = ['Train mse', 'train std', 'test mse', 'test std']
        ).T
    df.to_csv(report_path)



