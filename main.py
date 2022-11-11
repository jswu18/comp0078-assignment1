import os

import jax
import numpy as np
import pandas as pd

from src.constants import OUTPUTS_FOLDER
from src.models.helpers import phi_polynomial, phi_sin, train_test_split
from src.solutions import q1, q2_and_3, q4_and_5, q6, q7, q8

if __name__ == "__main__":
    jax.config.update("jax_platform_name", "cpu")

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
        figure_title="Question 1a: Regression",
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
    q2_and_3.a_i(
        number_of_sample_points,
        sigma,
        figure_title="Question 2ai: Noisy Samples",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2ai.png"),
    )
    q2_and_3.a_ii(
        number_of_sample_points,
        sigma,
        ks,
        figure_title="Question 2aii: Regression with Polynomial Basis",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2aii.png"),
    )
    q2_and_3.b(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        figure_title="Question 2b: Training Error vs Polynomial Dimension",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2b.png"),
        phi=phi_polynomial,
    )
    q2_and_3.c_and_d(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=1,
        figure_title="Question 2c: Test Error",
        figure_path=os.path.join(Q2_OUTPUT_FOLDER, "q2c.png"),
        phi=phi_polynomial,
    )
    q2_and_3.c_and_d(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=100,
        figure_title="Question 2d: Test Error",
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
    q2_and_3.b(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        figure_title="Question 3b: Training Error vs Polynomial Dimension",
        figure_path=os.path.join(Q3_OUTPUT_FOLDER, "q3b.png"),
        phi=phi_sin,
    )
    q2_and_3.c_and_d(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=1,
        figure_title="Question 3c: Test Error",
        figure_path=os.path.join(Q3_OUTPUT_FOLDER, "q3c.png"),
        phi=phi_sin,
    )
    q2_and_3.c_and_d(
        number_of_sample_points,
        sigma,
        ks=np.arange(1, 19),
        number_test_points=1000,
        number_of_trials=100,
        figure_title="Question 2d: Test Error",
        figure_path=os.path.join(Q3_OUTPUT_FOLDER, "q3d.png"),
        phi=phi_sin,
    )

    # Question 4 and 5
    Q4_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q4")
    Q5_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q5")

    if not os.path.exists(Q4_OUTPUT_FOLDER):
        os.makedirs(Q4_OUTPUT_FOLDER)

    if not os.path.exists(Q5_OUTPUT_FOLDER):
        os.makedirs(Q5_OUTPUT_FOLDER)

    data = pd.read_csv("data/Boston-filtered.csv")
    x = np.array(data)
    x, y = x[:, :-1], x[:, -1]
    x_train, y_train, x_test, y_test = train_test_split(x, y, 2 / 3)

    q4_and_5.all_parts(
        x_train,
        y_train,
        data_columns=list(data.columns),  # type: ignore
        figure_title_train="MSE Train",
        figure_title_test="MSE Test",
        figure_path_train=os.path.join(Q4_OUTPUT_FOLDER, "q4_train.png"),
        figure_path_test=os.path.join(Q4_OUTPUT_FOLDER, "q4_test.png"),
        gkrr_param_path=os.path.join(Q5_OUTPUT_FOLDER, "optimal_gkrr_parameters.csv"),
        contour_path=os.path.join(Q5_OUTPUT_FOLDER, "contour_plot"),
        report_path=os.path.join(Q5_OUTPUT_FOLDER, "regression_report.csv"),
    )  # type: ignore

    # Question 6
    Q6_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q6")
    if not os.path.exists(Q6_OUTPUT_FOLDER):
        os.makedirs(Q6_OUTPUT_FOLDER)
    number_of_points = 100
    v = 3
    n_dims = 2
    q6.all_parts(
        number_of_points,
        v,
        n_dims,
        figure_title=f"Question 6: h_(S={number_of_points},{v=})",
        figure_path=os.path.join(Q6_OUTPUT_FOLDER, "q6.png"),
    )

    # Question 7
    # Q7_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q7")
    # if not os.path.exists(Q7_OUTPUT_FOLDER):
    #     os.makedirs(Q7_OUTPUT_FOLDER)
    # v = 3
    # n_dims = 2
    # # actual params
    # k_values = np.arange(1, 50)
    # number_of_runs = 100
    # number_of_real_points = 100
    # number_of_train_points = 4000
    # number_of_test_points = 1000

    # # # testing params
    # # k_values = np.arange(1, 50)
    # # number_of_runs = 3
    # # number_of_real_points = 10
    # # number_of_train_points = 40
    # # number_of_test_points = 10

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

    # # Question 8
    # Q8_OUTPUT_FOLDER = os.path.join(OUTPUTS_FOLDER, "q8")
    # if not os.path.exists(Q8_OUTPUT_FOLDER):
    #     os.makedirs(Q8_OUTPUT_FOLDER)
    # v = 3
    # n_dims = 2
    # # actual params
    # k_values = np.arange(1, 50)
    # number_of_runs = 100
    # number_of_real_points = 100
    # number_of_train_points_vector = np.concatenate((np.array([100]), np.arange(500, 4500, 500)))
    # number_of_test_points = 1000

    # # # test params
    # # k_values = np.arange(1, 50)
    # # number_of_runs = 3
    # # number_of_real_points = 10
    # # number_of_train_points_vector = np.arange(50, 110, 10)
    # # number_of_test_points = 10

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
