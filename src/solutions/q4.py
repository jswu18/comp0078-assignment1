import numpy as np
import pandas as pd

from src.models.helpers import mean_squared_error, split_train_test_data
from src.models.regression import (
    LinearRegression,
    NaiveRegression,
)

PERFORMANCE_COLUMN_NAMES = [
    "Model",
    "Train Set MSE Mean",
    "Train Set MSE Standard Deviation",
    "Test Set MSE Mean",
    "Test Set MSE Standard Deviation",
]


def _compute_naive_performance(x, y, train_percentage: float, number_of_runs: int):
    train_mses = []
    test_mses = []
    model = NaiveRegression()
    for _ in range(number_of_runs):
        train_test_data = split_train_test_data(x, y, train_percentage)
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
    return pd.DataFrame(
        data=[
            [
                "Naive Regression",
                np.mean(train_mses),
                np.std(train_mses),
                np.mean(test_mses),
                np.std(test_mses),
            ]
        ],
        columns=PERFORMANCE_COLUMN_NAMES,
    )


def _compute_linear_performance(
    x, y, train_percentage: float, number_of_runs: int, model_name: str
):
    train_mses = []
    test_mses = []
    model = LinearRegression()
    for _ in range(number_of_runs):
        train_test_data = split_train_test_data(x, y, train_percentage)
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
    return pd.DataFrame(
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


def compute_q4_performance(
    df, feature_columns, target_column, train_percentage: float, number_of_runs: int
):
    y = df[[target_column]].values[:]

    df_naive_performance = _compute_naive_performance(
        df[[feature_columns[0]]].values[:], y, train_percentage, number_of_runs
    )

    df_linear_regression_performance_list: [pd.DataFrame] = []
    features_for_models = [[feature] for feature in feature_columns] + [feature_columns]
    model_names = [f"Linear Regression ({feature=})" for feature in feature_columns] + [
        "Linear Regression (all features)"
    ]
    for i, model_features in enumerate(features_for_models):
        x = df[model_features].values[:]
        df_linear_regression_performance_list.append(
            _compute_linear_performance(
                x, y, train_percentage, number_of_runs, model_name=model_names[i]
            )
        )
    df_linear_regression_performance = pd.concat(df_linear_regression_performance_list)
    df_performance = pd.concat([df_naive_performance, df_linear_regression_performance])
    return df_performance.set_index(PERFORMANCE_COLUMN_NAMES[0])


def all_parts(
    df,
    feature_columns,
    target_column,
    train_percentage: float,
    number_of_runs: int,
    performance_csv_path,
):
    df_performance = compute_q4_performance(
        df,
        feature_columns,
        target_column,
        train_percentage,
        number_of_runs,
    )
    df_performance.to_csv(performance_csv_path)
