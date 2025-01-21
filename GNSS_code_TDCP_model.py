import numpy as np
# model for GNSS code and TDCP, nonlinear
def h_GNSS_code_TDCP(x_u, x_s1, x_s2):
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
    h_x = np.vstack((rho, phase))  # shape should be (4kx1)

    # build jacobian
    # jacobian for pseudorange for 1st epoch
    jacobian_rho_1 = np.hstack(
        (-1 * (x_s1[0:3, :] - x_u1[0:3]).T / r_1, np.ones((r_1.size, 1)))
    )
    # jacobian for pseudorange for 2nd epoch
    jacobian_rho_2 = np.hstack(
        (-1 * (x_s2[0:3, :] - x_u2[0:3]).T / r_2, np.ones((r_2.size, 1)))
    )
    # create the space to store jaboian matrix for 1st and 2nd epoch
    jacobian_rho = np.zeros(
        (
            jacobian_rho_1.shape[0] + jacobian_rho_2.shape[0],
            jacobian_rho_1.shape[1] + jacobian_rho_2.shape[1],
        ),
        dtype=jacobian_rho_1.dtype,
    )
    # assign jacobian_rho_1 to even rows, 1st to 4th columns
    jacobian_rho[0::2, 0:4] = jacobian_rho_1
    # assign jacobian_rho_2 to add rows, 5th to last columns
    jacobian_rho[1::2, 4:] = jacobian_rho_2
    # assume phase measurement has the same model as pseudorange
    jacobian = np.vstack((jacobian_rho, jacobian_rho))  # shape should be (4kx8)

    return (h_x, jacobian)