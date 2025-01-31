import algebra as alg
import linear_least_squares as lls
import variables
import numpy as np
import scipy.stats as stats
from scipy.linalg import block_diag
from plot_skyplot import plot_skyplot


def criteria_to_continue(x):
    return not np.all(np.abs(x) <= variables.criteria)


def nonlinear_least_squares(h_model, y, R_model, x0, **dict_args_h_model):
    # h_model: function, that takes first an (n*1) vector as input,
    # and then takes NAMED parameters as specified in dict_args_h_model (see examples at bottom)
    # and returns an (m*1) vector and an (m*n) jacobian matrix

    # y: (m,1) vector (observations)

    # R_model: (m,m) covariance (invert of weight) matrix

    # x0: (n,1) vector: initial point

    # returns an (n,1) vector

    i = 1
    while True:
        print("iteration {}".format(i))
        hx0, Jac = h_model(x0, **dict_args_h_model)
        dx0 = lls.leastSquares(y - hx0, Jac, R_model)
        # x0[:, :] = x0[:, :] + dx0 # bug here??
        x0 = x0 + dx0
        i += 1

        # Check if iteration exceeds 10000
        if i > 3000:
            param1 = dict_args_h_model['x_s1']

            sat_pos = []

            for i in range(param1.shape[1]):  
                x, y, z = param1[0:3, i]  # 取出每一列的前 3 個元素 (x, y, z)
                # lat, lon, alt = ecef2lla(x, y, z)  # 轉換為 lat, lon, alt
                sat_pos.append([x, y, z])  # 將結果添加到列表中

            # 將結果轉換為 numpy 陣列
            sat_pos = np.array(sat_pos)


            print("Maximum iteration reached, plotting skyplot and breaking loop.")
            plot_skyplot(sat_pos, x0[0:3].reshape(-1))
            break

        if criteria_to_continue(dx0) is not True:
            print("done in {} iterations".format(i - 1))
            return x0


def toy_example_linear_ls():
    # check toy example, linear case
    def h_toy_example(x, H):
        result = H @ x
        jacobian = H
        return (result, jacobian)

    # x true for toy example
    x_true_toy = np.array([[1], [2]])
    # observation matrix
    H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1], [5, 9]])
    y = h_toy_example(x_true_toy, H)[0]
    y = y + np.random.randn(6, 1)
    # covariance matrix
    R = np.eye(6)
    # x0 for toy example
    x0_toy = np.array([[1], [2]])
    # LS
    estimate_result = nonlinear_least_squares(h_toy_example, y, R, x0_toy, H=H)
    print(f"x_true = \n{x_true_toy}")
    print(f"estimate = \n{estimate_result}")


def toy_example_non_linear_ls():
    # model for toy GNSS example, nonlinear
    def h_toy_GNSS(x_u, x_s):
        r = np.linalg.norm(x_s - x_u, axis=0).reshape(-1, 1)  # geometric range, (mx1)
        jacobian = -1 * (x_s - x_u).T / r  # each row is a unit vector from user to sat
        return (r, jacobian)

    # user position
    x_u_true = np.array([1, 2, 5]).reshape(-1, 1)
    # satellite position
    x_s = np.array(
        [[100, 0, 0, -100, 0, 0], [0, 0, 100, 0, 0, -100], [0, -100, 0, 0, 100, 0]]
    )
    y = h_toy_GNSS(x_u_true, x_s)[0]
    y = y + 0.01 * np.random.randn(y.shape[0], 1)  # add noise
    # add fault
    y[0] = y[0] + 10.5
    # covariance matrix, assume diagonal
    R = np.eye(x_s.shape[1])
    # x0 for start
    x0 = np.array([0, 0, 0]).reshape(-1, 1)
    # LS
    estimate_result = nonlinear_least_squares(h_toy_GNSS, y, R, x0, x_s=x_s)
    print(f"x_u_true = \n{x_u_true}")
    print(f"estimate = \n{estimate_result}")

    # calculate residual
    zhat_ls = y - h_toy_GNSS(estimate_result, x_s)[0]
    # calculate test statistics
    z = zhat_ls.T @ zhat_ls
    # calcuate threshold from inverse chi-squared cdf
    P_fa_set = 0.1
    T = stats.chi2.ppf(1 - P_fa_set, y.shape[0] - x0.shape[0])
    # compare test statistic with threshold
    if z <= T:
        fault_pred = 0
    else:
        fault_pred = 1

    print(f"Fault_pred: {fault_pred}")


def toy_GNSS_example_non_linear_ls_w_usr_clk_b():
    # model for toy GNSS example, nonlinear
    def h_toy_GNSS_w_usr_clk_b(x_u, x_s):
        r = np.linalg.norm(x_s[0:3, :] - x_u[0:3], axis=0).reshape(
            -1, 1
        )  # geometric range, (mx1)
        rho = r + (x_u[-1] - x_s[-1, :]).reshape(
            -1, 1
        )  # add clock bias to build pseudorange
        jacobian = np.hstack(
            (-1 * (x_s[0:3, :] - x_u[0:3]).T / r, np.ones((rho.size, 1)))
        )  # each row is a unit vector from user to sat
        return (rho, jacobian)

    # user position
    x_u_true = np.array([1, 2, 5, 1]).reshape(-1, 1)
    # satellite position
    x_s = np.array(
        [
            [100, 0, 0, -100, 0, 0],
            [0, 0, 100, 0, 0, -100],
            [0, -100, 0, 0, 100, 0],
            [1, 2, 3, 4, 5, 6],
        ]
    )
    y = h_toy_GNSS_w_usr_clk_b(x_u_true, x_s)[0]
    y = y + 0.01 * np.random.randn(y.shape[0], 1)  # add noise
    # covariance matrix, assume diagonal
    R = np.eye(x_s.shape[1])
    # x0 for start
    x0 = np.array([0, 0, 0, 0]).reshape(-1, 1)
    # LS
    estimate_result = nonlinear_least_squares(h_toy_GNSS_w_usr_clk_b, y, R, x0, x_s=x_s)
    print(f"x_u_true = \n{x_u_true}")
    print(f"estimate = \n{estimate_result}")


def toy_GNSS_example_non_linear_ls_TDCP():
    # model for GNSS code and TDCP, nonlinear
    def h_toy_GNSS_code_TDCP(x_u, x_s1, x_s2):
        # x_u1 is the user state at epoch 1, shape (4x1), [x1,y1,z1,tb1]^T
        # x_u2 is the user state at epoch 2, shape (4x1)
        x_u1 = x_u[0:4]
        x_u2 = x_u[4:]
        # x_s1 is the states of all k satellites at epoch 1, shape (4xk)
        # x_s2 is the states of all k satellites at epoch 2, shape (4xk)
        # geometric range between satellites and x_u1
        r_1 = np.linalg.norm(x_s1[0:3, :] - x_u1[0:3], axis=0).reshape(-1, 1)
        # geometric range between satellites and x_u2
        r_2 = np.linalg.norm(x_s2[0:3, :] - x_u2[0:3], axis=0).reshape(-1, 1)
        # pseudorange between satellites and user at epoch 1, add clock bias
        rho_1 = r_1 + (x_u1[-1] - x_s1[-1, :]).reshape(-1, 1)
        # pseudorange between satellites and user at epoch 2, add clock bias
        rho_2 = r_2 + (x_u2[-1] - x_s2[-1, :]).reshape(-1, 1)
        # stack rho_1 and rho_2 alternatively
        rho = np.empty(
            (rho_1.shape[0] + rho_2.shape[0], rho_1.shape[1]), dtype=rho_1.dtype
        )
        rho[0::2, :] = rho_1  # fill even rows with rows from rho_1
        rho[1::2, :] = rho_2  # fill add rowss with rows from rho_2
        # assume phase measurement has the same model as pseudorange
        phase = rho
        # concate rho and phase array vertically
        h_x = np.vstack((rho, phase))

        # build jacobian
        jacobian_rho_1 = np.hstack(
            (-1 * (x_s1[0:3, :] - x_u1[0:3]).T / r_1, np.ones((r_1.size, 1)))
        )
        jacobian_rho_2 = np.hstack(
            (-1 * (x_s2[0:3, :] - x_u2[0:3]).T / r_2, np.ones((r_2.size, 1)))
        )
        jacobian_rho = np.zeros(
            (
                jacobian_rho_1.shape[0] + jacobian_rho_2.shape[0],
                jacobian_rho_1.shape[1] + jacobian_rho_2.shape[1],
            ),
            dtype=jacobian_rho_1.dtype,
        )
        jacobian_rho[0::2, 0:4] = jacobian_rho_1
        jacobian_rho[1::2, 4:] = jacobian_rho_2
        # assume phase measurement has the same model as pseudorange
        jacobian = np.vstack((jacobian_rho, jacobian_rho))

        return (h_x, jacobian)

    # user states
    x_u_true = np.array([1, 2, 5, 1, 1, 2, 3, 1]).reshape(-1, 1)
    # satellite states at epoch 1
    x_s1 = np.array(
        [
            [100, 0, 0, -100, 0, 0],
            [0, 0, 100, 0, 0, -100],
            [0, -100, 0, 0, 100, 0],
            [1, 2, 3, 4, 5, 6],
        ]
    )
    # satellite states at epoch 2, need to be differnet from epoch 1
    # otherwise won't work...
    x_s2 = x_s1+5
    num_sats = x_s1.shape[1]  # number of satellites
    D = np.zeros((num_sats, 2 * num_sats), dtype=int)
    for i in range(num_sats):
        D[i, 2 * i] = -1
        D[i, 2 * i + 1] = 1

    A = block_diag(np.eye(2 * num_sats), D)
    y = h_toy_GNSS_code_TDCP(x_u_true, x_s1=x_s1, x_s2=x_s2)[0]
    y = y + 0.01 * np.random.randn(y.shape[0], 1)  # add noise
    # covariance matrix
    sigma_code = 3
    sigma_phase = 1
    R = block_diag(
        sigma_code**2 * np.eye(2 * x_s1.shape[1]),
        sigma_phase**2 * np.eye(2 * x_s1.shape[1]),
    )

    def h_A(x_u, x_s1, x_s2):
        hx, jacobian = h_toy_GNSS_code_TDCP(x_u=x_u, x_s1=x_s1, x_s2=x_s2)
        return (A @ hx, A @ jacobian)

    # x0 for start
    x0 = np.array([0, 0, 0, 0, 0, 0, 0, 0]).reshape(-1, 1)
    # LS
    estimate_result = nonlinear_least_squares(
        h_A, A @ y, A @ R @ A.T, x0, x_s1=x_s1, x_s2=x_s2
    )
    print(f"x_u_true = \n{x_u_true}")
    print(f"estimate = \n{estimate_result}")


if __name__ == "__main__":

    # check simple nonlinear case
    def cos(x):
        return (np.array([[np.cos(x[0, 0])]]), np.array([[-np.sin(x[0, 0])]]))

    x_true = np.array([[np.pi / 4]])
    R = np.eye(1)
    x0 = np.array([[np.pi / 2]])
    y = cos(x_true)[
        0
    ]  # no noise in observation, to see if solver returns exactly x_true, or not
    print("true = ", x_true)
    print("estimate = ", nonlinear_least_squares(cos, y, R, x0))

    # check linear case where function takes parameters that are not estimated, a and b here
    def h(x, a, b, c, d):
        result = np.array([[a * (x[0, 0]) + b], [c * (x[0, 0]) + d]])
        jacobian = np.array([[a], [c]])
        return (result, jacobian)

    a_test = 1.0
    b_test = 1.0
    c_test = 1.0
    d_test = 1.0
    additional_params = {"a": a_test, "b": b_test, "c": c_test, "d": d_test}
    x_true = np.array([[0.5]])
    y = h(x_true, a_test, b_test, c_test, d_test)[
        0
    ]  # no noise in observation, to see if solver returns exactly x_true, or not
    y = y + np.random.randn(y.shape[0], 1)
    R = np.eye(2)
    x0 = np.random.randn(1, 1)
    # two ways to pass the parameters to the function:
    print(
        "estimate = ",
        nonlinear_least_squares(h, y, R, x0, a=a_test, b=b_test, c=c_test, d=d_test),
    )
    print("estimate = ", nonlinear_least_squares(h, y, R, x0, **additional_params))

    # toy_example_linear_ls()
    # toy_example_non_linear_ls()
    # toy_GNSS_example_non_linear_ls_w_usr_clk_b()
    toy_GNSS_example_non_linear_ls_TDCP()
