from typing import List

import matplotlib.pyplot as plt
import numpy as np

from src.models.helpers import mean_squared_error, train_test_split
from src.models.regression_models import (
    MultipleLinearRegression,
    NaiveLinearRegression,
    RegressionModel,
    SingleLinearRegression,
)


def _regression_report(models, x, y, iters):
    mse_results = np.zeros((iters, len(models)))
    for i in range(iters):
        x_train, y_train, x_test, y_test = train_test_split(x, y, split=2 / 3)
        for j, model in enumerate(models):
            model.reset()
            mse_results[i, j] = mean_squared_error(
                model.fit_predict(x_train, y_train, x_test), y_test
            )
    return np.mean(mse_results, axis=0), np.std(mse_results, axis=0)


def all_parts(x_train, y_train, data_columns, figure_title, figure_path):
    models: List[RegressionModel] = []
    models += [SingleLinearRegression(idx) for idx in range(12)]
    models += [NaiveLinearRegression(), MultipleLinearRegression()]

    mses, stds = _regression_report(models, x_train, y_train, iters=20)

    axis_labels = data_columns[:12] + ["Naive", "MLR"]
    plt.figure(figsize=(15, 3))
    plt.bar(axis_labels, mses)
    plt.ylabel("MSE")
    plt.title(figure_title)
    plt.savefig(figure_path)
