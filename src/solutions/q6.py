import matplotlib.pyplot as plt
import numpy as np

from src.models.knn import get_p_h, knn_prediction


def all_parts(number_of_points, v, n_dims, figure_title, figure_path):
    p_h = get_p_h(number_of_points, n_dims)
    num_points_along_axis = 301
    grid_axis = np.linspace(0, 1, num_points_along_axis)
    x1, x2 = np.meshgrid(grid_axis, grid_axis)
    test_points = np.stack((x1, x2)).reshape(2, -1).T
    test_predictions = knn_prediction(p_h, test_points, k=v).reshape(
        num_points_along_axis, num_points_along_axis
    )

    fig, ax = plt.subplots(1, 1)
    fig.set_figwidth(10)
    fig.set_figheight(10)

    plt.imshow(np.flip(test_predictions, axis=0), extent=[0, 1, 0, 1])
    x, y = p_h
    idx_pos = np.where(y == 1)
    idx_neg = np.where(y == 0)

    plt.scatter(x[idx_pos, 0], x[idx_pos, 1], color="blue")
    plt.scatter(x[idx_neg, 0], x[idx_neg, 1], color="green")
    plt.title(figure_title)
    plt.savefig(figure_path)
