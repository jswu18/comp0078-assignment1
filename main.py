import os

import jax
import numpy as np
import pandas as pd

from src.constants import DATA_FOLDER, DEFAULT_SEED, OUTPUTS_FOLDER
from src.models.helpers import phi_polynomial, phi_sin
from src.solutions import q1, q2_and_3, q4, q5, q6, q7, q8

if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")
    jax.config.update("jax_enable_x64", True)

    if not os.path.exists(OUTPUTS_FOLDER):
        os.makedirs(OUTPUTS_FOLDER)

    # Question 1
    Q1_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q1")
    if not os.path.exists(Q1_OUTPUT_FOLDER):
        os.makedirs(Q1_OUTPUT_FOLDER)

    x = np.array([1, 2, 3, 4]).reshape(-1, 1)
    y = np.array([3, 2, 0, 5]).reshape(-1, 1)
    ks = [1, 2, 3, 4]
    q1.a(
        x,
        y,
        ks,
        figure_title="Regression",
        figure_path=os.path.join(Q1_OUTPUT_FOLDER, "q1a.png"),
    )
    q1.b(x, y, ks, table_path=os.path.join(Q1_OUTPUT_FOLDER, "q1b.csv"))
    q1.c(x, y, ks, table_path=os.path.join(Q1_OUTPUT_FOLDER, "q1c.csv"))

    # Question 2
    Q2_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q2")
    if not os.path.exists(Q2_OUTPUT_FOLDER):
        os.makedirs(Q2_OUTPUT_FOLDER)
    number_of_sample_points = 30
    sigma = 0.07
    ks = np.array([2, 5, 10, 14, 18])
    np.random.seed(DEFAULT_SEED)
    q2_and_3.a_i(
        number_of_sample_points,
        sigma,
        figure_title="Noisy Samples",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2ai.png"),
    )
    q2_and_3.a_ii(
        number_of_sample_points,
        sigma,
        ks,
        figure_title="Regression with Polynomial Basis",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2aii.png"),
    )
    q2_and_3.bcd(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=1,
        figure_title="Error vs Polynomial Dimension",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2bc.png"),
        phi=phi_polynomial,
    )
    q2_and_3.bcd(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=100,
        figure_title="TError vs Polynomial Dimension",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2d.png"),
        phi=phi_polynomial,
    )

    # Question 3
    Q3_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q3")
    if not os.path.exists(Q3_OUTPUT_FOLDER):
        os.makedirs(Q3_OUTPUT_FOLDER)

    number_of_sample_points = 30
    sigma = 0.07
    ks = np.arange(1, 19)
    q2_and_3.bcd(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=1,
        figure_title="Error vs Number of Sin Bases",
        figure_path=os.path.join(Q3_OUTPUT_FOLDER, "q3bc.png"),
        phi=phi_sin,
    )
    q2_and_3.bcd(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=100,
        figure_title="Error vs Number of Sin Bases",
        figure_path=os.path.join(Q3_OUTPUT_FOLDER, "q3d.png"),
        phi=phi_sin,
    )

    df = pd.read_csv(os.path.join(DATA_FOLDER, "Boston-filtered.csv"))
    feature_columns = list(df.columns[:-1])
    target_column = df.columns[-1]

    # Question 4
    np.random.seed(DEFAULT_SEED)
    Q4_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q4")
    train_percentage = 2 / 3
    number_of_runs = 20
    q4_performance_csv_path = os.path.join(Q4_OUTPUT_FOLDER, "performance.csv")
    q4.all_parts(
        df,
        feature_columns,
        target_column,
        train_percentage,
        number_of_runs,
        performance_csv_path=q4_performance_csv_path,
    )

    # Question 5
    np.random.seed(DEFAULT_SEED)
    Q5_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q5")
    train_percentage = 2 / 3
    q5.abc(
        df,
        feature_columns,
        target_column,
        train_percentage,
        number_of_folds=5,
        log_2_gammas=np.arange(-40, -25),
        log_2_sigmas=np.arange(7, 13.5, 0.5),
        optimal_params_csv_path=os.path.join(
            Q5_OUTPUT_FOLDER, "q5a-optimal-params.csv"
        ),
        performance_csv_path=os.path.join(Q5_OUTPUT_FOLDER, "q5c-performance.csv"),
        mse_figure_path=os.path.join(Q5_OUTPUT_FOLDER, "q5b-cross-valid-error.png"),
    )
    q5.d(
        df,
        feature_columns,
        target_column,
        train_percentage,
        number_of_runs=20,
        number_of_folds=5,
        log_2_gammas=np.arange(-40, -25),
        log_2_sigmas=np.arange(7, 13.5, 0.5),
        q4_performance_csv_path=q4_performance_csv_path,
        performance_csv_path=os.path.join(Q5_OUTPUT_FOLDER, "q5d-performance.csv"),
    )

    # # Question 6
    # Q6_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q6")
    # if not os.path.exists(Q6_OUTPUT_FOLDER):
    #     os.makedirs(Q6_OUTPUT_FOLDER)
    # number_of_points = 100
    # v = 3
    # n_dims = 2
    # q6.all_parts(
    #     number_of_points,
    #     v,
    #     n_dims,
    #     figure_title=f"Question 6: h_(S={number_of_points},{v=})",
    #     figure_path=os.path.join(Q6_OUTPUT_FOLDER, "q6.png"),
    # )

    # # Question 7
    # Q7_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q7")
    # if not os.path.exists(Q7_OUTPUT_FOLDER):
    #     os.makedirs(Q7_OUTPUT_FOLDER)
    # v = 3
    # n_dims = 2
    # k_values = np.arange(1, 50)
    # number_of_runs = 100
    # number_of_real_points = 100
    # number_of_train_points = 4000
    # number_of_test_points = 1000
    #
    # q7.all_parts(
    #     k_values,
    #     v,
    #     n_dims,
    #     number_of_runs,
    #     number_of_real_points,
    #     number_of_train_points,
    #     number_of_test_points,
    #     figure_title="Question 7: Protocol A",
    #     figure_path=os.path.join(Q7_OUTPUT_FOLDER, "q7.png"),
    # )
    #
    # # Question 8
    # Q8_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q8")
    # if not os.path.exists(Q8_OUTPUT_FOLDER):
    #     os.makedirs(Q8_OUTPUT_FOLDER)
    # v = 3
    # n_dims = 2
    # k_values = np.arange(1, 50)
    # number_of_runs = 100
    # number_of_real_points = 100
    # number_of_train_points_vector = np.concatenate(
    #     (np.array([100]), np.arange(500, 4500, 500))
    # )
    # number_of_test_points = 1000
    #
    # q8.all_parts(
    #     k_values,
    #     v,
    #     n_dims,
    #     number_of_runs,
    #     number_of_real_points,
    #     number_of_train_points_vector,
    #     number_of_test_points,
    #     figure_title="Question 8: Protocol B",
    #     figure_path=os.path.join(Q8_OUTPUT_FOLDER, "q8.png"),
    # )
