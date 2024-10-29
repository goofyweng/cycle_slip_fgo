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


def cal_test_statistic(y: np.array, H: np.array):
    # calculate pseudo inverse
    S = pseudo_inv(H)
    # least square
    x_hat = S @ y
    # calculate residual
    z_hat_ls = y - H @ x_hat
    # test statistics
    z = z_hat_ls.T @ z_hat_ls

    return z.item()


def cal_chi_square_centrality(mu, H):
    # calculate the non-centrality for chi-square
    centrality = mu.T @ (np.eye(H.shape[0]) - H @ pseudo_inv(H)) @ mu

    return centrality.item()


def cal_test_statistic_H0(x_true: np.array = None, H: np.array = None):
    if x_true is None:
        x_true = np.array([[1], [2]])

    if H is None:
        H = H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1]])
    # Generate a 4x1 column vector of normally distributed random variables
    epsilon = np.random.randn(H.shape[0], 1)
    # observation vector, measurement
    y = H @ x_true + epsilon
    # calculate test statistics
    z = cal_test_statistic(y, H)

    return z


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

    # test statistics
    z = cal_test_statistic(y, H)

    # non-centrality for chi-square
    centrality = cal_chi_square_centrality(mu, H)

    return z, centrality


def cal_test_statistic_H1_mc(
    x_true: np.array = None, H: np.array = None, mu: np.array = None, num_runs: int = 1
):
    z_array = np.zeros((num_runs,))
    for i in range(num_runs):
        z_array[i], centrality = cal_test_statistic_H1(x_true, H, mu)

    return z_array, centrality


def toy_example_fault_detection():
    """
    A toy example for fault detection. The test statistics is calculated in Monte Carlo simulation.
    The distiburition of the test statistics is drawn and compared with theroitical chi-square disctribution.
    """
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


def toy_example_fault_identification():
    """
    This toy example assume fault detection is performed and we know the fault exist in observation vector.
    Here we only consider single fault case. We then move on to the second step: try to identify which observation
    is contaminated by the fault.
    """
    # set up
    # state true value
    x_true = np.array([[1], [2]])
    # observation matrix
    H = np.array([[1, 0], [0, 1], [1, 1], [1, 3], [-1, 1], [1, -2]])
    # number of states
    n = H.shape[1]
    # number of measurements
    m = H.shape[0]
    # setup e matrix which contains all the e_i vector in each column, i=1, 2,..., 6
    e_matrix = np.eye(6)
    # the value of fault
    mu = 7
    # number of Monte Carlo simulations
    mc = 2000

    # store the calculated test statistic results,
    z_result = np.zeros((e_matrix.shape[0], mc, e_matrix.shape[1]))
    # calculate the test statistic for different e_j
    for j in range(e_matrix.shape[0]):
        for mc_iter in range(mc):
            # Generate a column vector of normally distributed random variables
            epsilon = np.random.randn(H.shape[0], 1)
            # observation vector, here we put the fault in the j-th observation value
            y = H @ x_true + epsilon + mu * e_matrix[:, j].reshape(-1, 1)
            # for different e_i, calculate the test stastics
            for i in range(e_matrix.shape[1]):
                # append the e_i column vector to the observation matrix, H, to generate H_i
                H_i = np.column_stack((H, e_matrix[:, i]))
                # store the result
                z_result[j, mc_iter, i] = cal_test_statistic(y, H_i)

    # Create a figure and axis
    fig, ax = plt.subplots(2, 3, figsize=(12, 8))
    # Plot results for different e_j
    for j in range(6):
        row, col = divmod(j, 3)  # Calculate row and column index
        # plot the histogram for the test statistic calculated by different e_i
        # we expect when e_j = e_i, the distribution of test statistic should be a central chi-square distribution
        for i in range(e_matrix.shape[1]):
            ax[row, col].hist(
                z_result[j, :, i],
                bins=30,
                density=True,
                # color="blue",
                alpha=0.5,
                edgecolor="black",
                label=f"$\\text{{z}}_{{{{H}}_{i}}}$",
            )
        # Parameters
        x_limit = 40  # Limit for x-axis
        # Call the function to plot the central chi-squared distribution
        # When e_j = e_i, the distribution of test statistic z_i should be a central chi-square distribution
        plot_non_central_chi2(ax[row, col], m - n - 1, 0, x_limit)
        ax[row, col].set_xlim([0, x_limit])
        ax[row, col].legend()
        ax[row, col].set_title(f"$e_j = e_{j}$")

    fig2, ax2 = plt.subplots()
    for i in range(e_matrix.shape[1]):
        ax2.hist(
            z_result[0, :, i],
            bins=30,
            density=True,
            # color="blue",
            alpha=0.5,
            edgecolor="black",
            label=f"$\\text{{z}}_{{{{H}}_{i}}}$",
        )
        # Call the function to plot the central chi-squared distribution
        # When e_j = e_i, the distribution of test statistic z_i should be a central chi-square distribution
    plot_non_central_chi2(ax2, m - n - 1, 0, x_limit)
    ax2.legend()
    ax2.set_title(f"$e_j = e_{0}$")
    
    # Show the plot
    fig.tight_layout()
    fig2.tight_layout()
    plt.show()


if __name__ == "__main__":
    # toy_example_fault_detection()
    toy_example_fault_identification()
