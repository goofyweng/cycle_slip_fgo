import algebra as alg
import linear_least_squares as lls
import variables
import numpy as np


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
    x_true_toy = np.array([[10], [20]])
    # observation matrix
    H = np.array([[1, 0], [0, 1], [1, 1], [2, 3], [-1, 1], [5, 9]])
    y = h_toy_example(x_true_toy, H)[0]
    # covariance matrix
    R = np.eye(H.shape[0])
    # x0 for toy example
    x0_toy = np.array([[12], [25]])
    # LS
    estimate_result = nonlinear_least_squares(h_toy_example, y, R, x0_toy, H=H)
    print(f"x_true = \n{x_true_toy}")
    print(f"estimate = \n{estimate_result}")


def toy_example_non_linear_ls():
    # model for toy GNSS example, nonlinear
    def h_toy_GNSS(x_u, x_s):
        r = np.linalg.norm(x_s - x_u, axis=0).reshape(-1, 1)  # geometric range, (mx1)
        jacobian = -1 * (x_s - x_u).T / r  # each row is a unit vector from user to sat
        return (r.reshape(-1, 1), jacobian)

    # user position
    x_u_true = np.array([1, 2, 5]).reshape(-1, 1)
    # satellite position
    x_s = np.array(
        [[100, 0, 0, -100, 0, 0], [0, 0, 100, 0, 0, -100], [0, -100, 0, 0, 100, 0]]
    )
    y = h_toy_GNSS(x_u_true, x_s)[0]
    # y = y + np.random.randn(y.shape[0], 1) # add noise
    # covariance matrix, assume diagonal
    R = np.eye(x_s.shape[1])
    # x0 for start
    x0 = np.array([0, 0, 0]).reshape(-1, 1)
    # LS
    estimate_result = nonlinear_least_squares(h_toy_GNSS, y, R, x0, x_s=x_s)
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
        jacobian = np.array([[a], [b]])
        return (result, jacobian)

    a_test = 1.0
    b_test = 1.0
    c_test = 1.0
    d_test = 1.0
    additional_params = {"a": a_test, "b": b_test, "c": c_test, "d": d_test}
    x_true = np.array([[0.5]])
    y = h(x_true, a_test, b_test, c_test, d_test)[0]  # no noise in observation, to see if solver returns exactly x_true, or not
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
    toy_example_non_linear_ls()
