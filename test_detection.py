import numpy as np
import matplotlib.pyplot as plt
from plot_dist import plot_non_central_chi2


def pseudo_inv(H: np.array):
    """
    Calculate the pseudo inverse, S, from the observation matrix H
    If H has the shape (m x n), then S has the shape (n x m)
    """
    # check if H is full rank
    rank = np.linalg.matrix_rank(H)
    assert rank == min(H.shape), "Input matrix is not full rank."
    S = np.linalg.inv(H.T @ H) @ H.T
    return S

def cal_test_statistic_H0(x_true: np.array = None, H: np.array = None):
    if x_true is None:
        x_true = np.array([[1], [2]])

    if H is None:
        H = H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1]])
    # Generate a 4x1 column vector of normally distributed random variables
    epsilon = np.random.randn(H.shape[0], 1)
    # observation vector, measurement
    y = H @ x_true + epsilon

    # Least Square
    S = pseudo_inv(H)
    x_hat = S @ y

    # residual
    z_hat_ls = y - H @ x_hat

    # test statistics
    z = z_hat_ls.T @ z_hat_ls

    return z.item()


def cal_test_statistic_H0_mc(
    x_true: np.array = None, H: np.array = None, num_runs: int = 1
):
    z_array = np.zeros((num_runs,))
    for i in range(num_runs):
        z_array[i] = cal_test_statistic_H0(x_true, H)

    return z_array


def cal_test_statistic_H1(
    x_true: np.array = None, H: np.array = None, mu: np.array = None
):
    if x_true is None:
        x_true = np.array([[1], [2]])

    if H is None:
        H = H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1]])

    if mu is None:
        mu = np.array([[3], [0], [0], [0], [0]])

    # Generate a 4x1 column vector of normally distributed random variables
    epsilon = np.random.randn(H.shape[0], 1)
    # observation vector, measurement
    y = H @ x_true + epsilon + mu

    # Least Square
    S = pseudo_inv(H)
    x_hat = S @ y

    # residual
    z_hat_ls = y - H @ x_hat

    # test statistics
    z = z_hat_ls.T @ z_hat_ls

    # non-centrality for chi-square
    centrality = mu.T @ (np.eye(H.shape[0]) - H @ S) @ mu

    return z.item(), centrality.item()


def cal_test_statistic_H1_mc(
    x_true: np.array = None, H: np.array = None, mu: np.array = None, num_runs: int = 1
):
    z_array = np.zeros((num_runs,))
    for i in range(num_runs):
        z_array[i], centrality = cal_test_statistic_H1(x_true, H, mu)

    return z_array, centrality




if __name__ == "__main__":
    # set up
    # state true value
    x_true = np.array([[1], [2]])
    # observation matrix
    H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1], [5, 9]])
    # number of states
    n = H.shape[1]
    # number of measurements
    m = H.shape[0]
    # number of MonteCarlo runs
    n_mc = 5000
    # calculate test statistics in H0
    z_H0 = cal_test_statistic_H0(x_true, H)
    # calculate test statistics in H0, MonteCarlo
    z_H0_mc = cal_test_statistic_H0_mc(x_true, H, n_mc)

    mu = np.array([[3], [0], [0], [0], [0], [0]])
    # calculate test statistics in H1
    z_H1, centrality = cal_test_statistic_H1(x_true, H, mu=mu)
    # calculate test statistics in H1, MonteCarlo
    z_H1_mc, centrality = cal_test_statistic_H1_mc(x_true, H, mu, n_mc)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Parameters
    x_limit = 40  # Limit for x-axis

    # Call the function to plot the non-central chi-squared distribution
    plot_non_central_chi2(ax, m - n, 0, x_limit)
    plot_non_central_chi2(ax, m - n, centrality, x_limit)
    # draw_vertical_line(ax, z_H0, color="r", label=f"z_H0={z_H0:.4f}")
    # draw_vertical_line(ax, z_H1, color="g", label=f"z_H1={z_H1:.4f}")

    # Plot a histogram of the data on the specific axis
    ax.hist(
        z_H0_mc,
        bins=30,
        density=True,
        color="blue",
        alpha=0.5,
        edgecolor="black",
        label="z_H0",
    )
    ax.hist(
        z_H1_mc,
        bins=30,
        density=True,
        color="orange",
        alpha=0.5,
        edgecolor="black",
        label="z_H1",
    )
    ax.legend()

    # Show the plot
    plt.show()
