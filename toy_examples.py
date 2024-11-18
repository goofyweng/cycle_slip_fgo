import numpy as np
import matplotlib.pyplot as plt
from plot_dist import plot_non_central_chi2, draw_vertical_line
import scipy.stats as stats
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


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
        label=f"$\\text{{z}}_{{{{H}}_{0}}}$",
    )
    ax.hist(
        z_H1_mc,
        bins=30,
        density=True,
        color="orange",
        alpha=0.5,
        edgecolor="black",
        label=f"$\\text{{z}}_{{{{H}}_{1}}}$",
    )
    ax.legend()

    # Show the plot
    plt.show()


def toy_example_fault_detection_check_probability_of_false_alarm():
    """
    This toy example performs fault detection. Check if the
    Probability of false alarm, P_fa, is asyumptotically close
    to the set value.
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
    n_mcs = np.arange(5, 5001, 50)
    # error vector
    mu = np.array([[3], [0], [0], [0], [0], [0]])
    # define the threshold T, by the probability of false alarm
    P_fa_set = 0.1
    T = stats.chi2.ppf(1 - P_fa_set, m - n)
    # store calculate P_fa
    P_fa_result = np.zeros(n_mcs.shape)
    idx = 0
    for n_mc in n_mcs:
        # calculate test statistics in H0 (true no fault), MonteCarlo
        z_H0_mc = cal_test_statistic_H0_mc(x_true, H, n_mc)
        # calculate test statistics in H1, MonteCarlo
        z_H1_mc, centrality = cal_test_statistic_H1_mc(x_true, H, mu, n_mc)
        # number of False Positive
        n_FP = np.sum(z_H0_mc > T)
        # number of True Negative
        n_TN = z_H0_mc.size - n_FP
        # number of True Positive
        n_TP = np.sum(z_H1_mc > T)
        # number of False Negative
        n_FN = z_H1_mc - n_TP
        # Probability of false alarm
        P_fa = n_FP / (n_FP + n_TN)
        # Probability of detection
        P_det = n_TP / (n_TP + n_FN)
        # store P_fa
        P_fa_result[idx] = P_fa
        idx = idx + 1

    fig, ax = plt.subplots()
    ax.plot(n_mcs, P_fa_result, label="Calculated")
    ax.axhline(P_fa_set, color="red", linestyle="--", linewidth=2, label="Set value")
    ax.set_title("Probability of false alarm")
    ax.set_xlabel("Number of detections")
    ax.set_ylabel("$P_{fa}$")
    ax.legend()
    plt.show()


def toy_example_fault_detection_check_probability_of_detection():
    """
    This toy example performs fault detection. Check if the
    probability of detection increase as the fault amplitude increase.
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
    n_mc = 2000
    # error amplitude
    mu_value = np.arange(0, 11, 0.5)
    # define the threshold T, by the probability of false alarm
    P_fa_set = 0.1
    T = stats.chi2.ppf(1 - P_fa_set, m - n)
    # store calculate P_det
    P_det_result = np.zeros(mu_value.shape)
    idx = 0
    for mu_ in mu_value:
        # error vector
        mu = np.array([[mu_], [0], [0], [0], [0], [0]])
        # calculate test statistics in H1, MonteCarlo
        z_H1_mc, centrality = cal_test_statistic_H1_mc(x_true, H, mu, n_mc)
        # number of True Positive
        n_TP = np.sum(z_H1_mc > T)
        # number of False Negative
        n_FN = z_H1_mc.size - n_TP
        # Probability of detection
        P_det = n_TP / (n_TP + n_FN)
        # store P_fa
        P_det_result[idx] = P_det
        idx = idx + 1

    fig, ax = plt.subplots()
    ax.plot(mu_value, P_det_result, label="$P_{detection}$")
    ax.set_title("Probability of detection")
    ax.set_xlabel("Amplitude of $\mu$")
    ax.set_ylabel("$P_{detection}$")
    ax.legend()
    plt.show()


def toy_example_fault_identification_mc():
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
                # label=f"$\\text{{z}}_{{{{H}}_{i}}}$",
                label=f"$\\text{{z}}_{i}$",
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
            # label=f"$\\text{{z}}_{{{{H}}_{i}}}$",
            label=f"$\\text{{z}}_{i}$",
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


def toy_example_fault_identification_confusion_matrix():
    """
    This toy example assume fault detection is performed and we know the fault exist in observation vector.
    Here we only consider single fault case. We then move on to the second step: try to identify which observation
    is contaminated by the fault. We use the cauclated test statistics, z_i, as input to the central
    chi-squared distribution to find the location of the fault in measruement. The result is visiualized
    with confusion matrix.
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
    e_matrix = np.eye(H.shape[0])
    # the value of fault
    mu = 5
    # degree of freedom
    dof = m - n - 1
    # centrality of chi-squared dist.
    nc = 0
    # number of Monte Carlo runs
    mc = 2000

    cm_result_mc = np.zeros(e_matrix.shape)
    for mc_ in range(mc):
        # store the calculated test statistics
        z_result = np.zeros(e_matrix.shape)
        # Generate a column vector of normally distributed random variables
        epsilon = np.random.randn(H.shape[0], 1)
        # calculate the test statistic for differnet e_j
        for j in range(e_matrix.shape[0]):
            # observation vector, here we put the fault in the j-th observation value
            y = H @ x_true + epsilon + mu * e_matrix[:, j].reshape(-1, 1)
            # for different e_i, calculate the test statistics
            for i in range(e_matrix.shape[1]):
                # append the e_i column vector to the observation matrix, H, to generate H_i
                H_i = np.column_stack((H, e_matrix[:, i]))
                # store the result
                z_result[j, i] = cal_test_statistic(y, H_i)
        # get the value of chi-squared pdf with z_i as input
        # here each row represent different j value, i.e. differnece true location of the fault
        chi2_pdf_value = np.array([stats.ncx2.pdf(z_i, dof, nc) for z_i in z_result])
        # find i_hat by finding the index of the maximum value of the value of chi-squared pdf in each row
        i_hat = np.argmax(chi2_pdf_value, axis=1)
        # the true fault location
        i_true = np.argmax(e_matrix, axis=0)
        # build confusion matrix
        cm_result = confusion_matrix(i_true, i_hat)
        # store the confusion matrix result for MonteCarlo simulation
        cm_result_mc += cm_result

    cm_result_mc = cm_result_mc / np.sum(cm_result_mc) * 100
    # visiualize confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_result_mc)
    disp.plot()
    disp.ax_.set_xticklabels([f"$e_{i}$" for i in range(m)])
    disp.ax_.set_yticklabels([f"$e_{i}$" for i in range(m)])
    disp.ax_.set_xlabel("Predicted error, $e_i$")
    disp.ax_.set_ylabel("True error, $e_j$")
    plt.show()


def toy_example_pobability_of_false_alarm():
    # set up
    # state true value
    x_true = np.array([[1], [2]])
    # observation matrix
    H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1], [5, 9]])
    # number of states
    n = H.shape[1]
    # number of measurements
    m = H.shape[0]
    # calculate test statistics in H0
    z_H0 = cal_test_statistic_H0(x_true, H)

    mu = np.array([[3], [0], [0], [0], [0], [0]])
    # calculate test statistics in H1
    z_H1, centrality = cal_test_statistic_H1(x_true, H, mu=mu)

    # Create a figure and axis
    fig, ax = plt.subplots()

    # Parameters
    x_limit = 40  # Limit for x-axis

    # Call the function to plot the non-central chi-squared distribution
    plot_non_central_chi2(ax, m - n, 0, x_limit)
    plot_non_central_chi2(ax, m - n, centrality - 3, x_limit)
    plot_non_central_chi2(ax, m - n, centrality, x_limit)

    # Generate x values and the PDF for a chi-squared distribution
    x = np.linspace(0, x_limit, 1000)
    # Calculate the PDF for the non-central chi-squared distribution
    pdf_h0 = stats.ncx2.pdf(x, m - n, 0)
    pdf_h1 = stats.ncx2.pdf(x, m - n, centrality)
    # The threshold
    T = 6.1

    draw_vertical_line(ax, T, "red", "T")

    ax.fill_between(
        x, pdf_h0, where=(x >= T), color="blue", alpha=0.5, label="$P_{fa}$"
    )
    # ax.fill_between(x, pdf_h1, where=(x >= T), color="yellow", alpha=0.5, label="P_d")
    ax.legend()
    ax.set_ylim([0, 0.2])
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    # toy_example_fault_detection()
    # toy_example_fault_identification_mc()
    # toy_example_pobability_of_false_alarm()
    # toy_example_fault_identification_confusion_matrix()
    toy_example_fault_detection_check_probability_of_false_alarm()
    toy_example_fault_detection_check_probability_of_detection()
