import numpy as np
from scipy.linalg import block_diag


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
    # pseudorange between satellites and user at epoch 1, add user clock bias. 
    # The pseudorange is already corrected for satellite clock bias, ionsphere, troposhpere, etc, 
    # see 'correct_prx_data.py'
    rho_1 = r_1 + (x_u1[-1]).reshape(-1, 1)
    # pseudorange between satellites and user at epoch 2, add user clock bias
    rho_2 = r_2 + (x_u2[-1]).reshape(-1, 1)
    # stack rho_1 and rho_2 alternatively
    rho = np.empty((rho_1.shape[0] + rho_2.shape[0], rho_1.shape[1]), dtype=rho_1.dtype)
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


def build_h_A(data, epoch):
    """
    This function build the observation model, 'h(x)', for pseudorange and carrier 
    phase measurements using two consecutive epochs, and also the 'A' matrix to 
    build the TDCP observation model. A 'h_A(x)' function is returned which is 
    equivalent to A @ h(x).
    ==========
    Input:
    data: a pd.DataFrame contains corrected prx data, see correct_prx_data.py.
    epoch: the pair of chosen consecutive epochs (epoch t, and t+1) in data.
    ==========
    Output:
    common_sats: commen satellites prn in chosen epochs.
    sat_coor1: satellites ECEF coordinate in epoch t.
    sat_coor2: satellites ECEF coordinate in epoch t+1.
    sat_clock_bias1: satellites clock bias in epoch t.
    sat_clock_bias2: satellites clock bias in epoch t+1.
    y: vertically stacked observation vector which following the following pattern,
        [
        C_obs_m_corrected between user and satellite 1 at epoch t,
        C_obs_m_corrected between user and satellite 1 at epoch t+1,
        ...
        C_obs_m_corrected between user and satellite k at epoch t,
        C_obs_m_corrected between user and satellite k at epoch t+1,
        L_obs_m_corrected between user and satellite 1 at epoch t,
        L_obs_m_corrected between user and satellite 1 at epoch t+1,
        ...
        L_obs_m_corrected between user and satellite k at epoch t,
        L_obs_m_corrected between user and satellite k at epoch t+1,
        ]
        with shape (4k x 1)
    A: matrix used to build TDCP Factor Graph, with shape (3k x 4k)
    h_A: TDCP observation model, A @ h(x)
    sat_ele_rad: satellite elevation in radians, matching the same prn order in y
    """

    # Choose the data from two chosen epoch
    epoch_data1 = data[data["time_of_reception_in_receiver_time"] == epoch[0]]
    epoch_data2 = data[data["time_of_reception_in_receiver_time"] == epoch[1]]

    if epoch_data1.empty or epoch_data2.empty:
        print(f"No data for one of the epochs {epoch}")
        return None

    # find the common sats appear in both epochs
    sat_epoch1 = set(epoch_data1["prn"])
    sat_epoch2 = set(epoch_data2["prn"])
    common_sats = sorted(sat_epoch1 & sat_epoch2)  # find intersection and sort

    if not common_sats:
        print(f"No common satellites found between epochs {epoch}")
        return None

    # filter only the data from the common sats
    epoch_data1 = epoch_data1[epoch_data1["prn"].isin(common_sats)].set_index("prn")
    epoch_data2 = epoch_data2[epoch_data2["prn"].isin(common_sats)].set_index("prn")

    sat_coor1 = epoch_data1.loc[
        common_sats, ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]
    ].values
    sat_coor2 = epoch_data2.loc[
        common_sats, ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]
    ].values
    sat_clock_bias1 = epoch_data1.loc[common_sats, "sat_clock_offset_m"].values
    sat_clock_bias2 = epoch_data2.loc[common_sats, "sat_clock_offset_m"].values
    C_obs_m1 = epoch_data1.loc[common_sats, "C_obs_m_corrected"].values
    C_obs_m2 = epoch_data2.loc[common_sats, "C_obs_m_corrected"].values
    L_obs_cycles1 = epoch_data1.loc[common_sats, "L_obs_m_corrected"].values
    L_obs_cycles2 = epoch_data2.loc[common_sats, "L_obs_m_corrected"].values
    sat_ele_rad1 = np.deg2rad(epoch_data1.loc[common_sats, "sat_elevation_deg"].values)
    sat_ele_rad2 = np.deg2rad(epoch_data2.loc[common_sats, "sat_elevation_deg"].values)

    # reorder sat_ele_rad
    sat_ele_rad = np.empty((2 * len(common_sats), 1))
    sat_ele_rad[0::2] = sat_ele_rad1.reshape(-1, 1)
    sat_ele_rad[1::2] = sat_ele_rad2.reshape(-1, 1)

    # reorder code obs
    code_obs = np.empty((2 * len(common_sats), 1))
    code_obs[0::2] = C_obs_m1.reshape(-1, 1)  # even row epoch1
    code_obs[1::2] = C_obs_m2.reshape(-1, 1)  # odd row epoch2

    # reorder cycle obs
    cycle_obs = np.empty((2 * len(common_sats), 1))
    cycle_obs[0::2] = L_obs_cycles1.reshape(-1, 1)
    cycle_obs[1::2] = L_obs_cycles2.reshape(-1, 1)

    # upper part is Code Obs from common sats for both epoch, and low part the Cycle Obs
    y = np.concatenate([code_obs, cycle_obs], axis=0)

    # build D matrix in papper
    num_sats = sat_coor1.shape[0]  # number of satellites
    D = np.zeros((num_sats, 2 * num_sats), dtype=int)
    for i in range(num_sats):
        D[i, 2 * i] = -1
        D[i, 2 * i + 1] = 1

    # build A matrix
    A = block_diag(np.eye(2 * num_sats), D)

    # build TDCP model with A matrix
    def h_A(x_u, x_s1, x_s2):
        hx, jacobian = h_GNSS_code_TDCP(x_u=x_u, x_s1=x_s1, x_s2=x_s2)
        return (A @ hx, A @ jacobian)

    return (
        common_sats,
        sat_coor1,
        sat_coor2,
        sat_clock_bias1,
        sat_clock_bias2,
        y,
        A,
        h_A,
        sat_ele_rad,
    )


def build_h_A_num_sats(data, epoch, num_sats):
    """
    Similar function as build_h_A but only keep the data for desired number of satellites.
    """

    # Choose the data from two chosen epoch
    epoch_data1 = data[data["time_of_reception_in_receiver_time"] == epoch[0]]
    epoch_data2 = data[data["time_of_reception_in_receiver_time"] == epoch[1]]

    if epoch_data1.empty or epoch_data2.empty:
        print(f"No data for one of the epochs {epoch}")
        return None

    # find the common sats appear in both epochs
    sat_epoch1 = set(epoch_data1["prn"])
    sat_epoch2 = set(epoch_data2["prn"])
    common_sats = sorted(sat_epoch1 & sat_epoch2)  # find intersection and sort
    common_sats = common_sats[
        :num_sats
    ]  # keep only the first desired number of satellites

    if not common_sats:
        print(f"No common satellites found between epochs {epoch}")
        return None

    # filter only the data from the common sats
    epoch_data1 = epoch_data1[epoch_data1["prn"].isin(common_sats)].set_index("prn")
    epoch_data2 = epoch_data2[epoch_data2["prn"].isin(common_sats)].set_index("prn")

    sat_coor1 = epoch_data1.loc[
        common_sats, ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]
    ].values
    sat_coor2 = epoch_data2.loc[
        common_sats, ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]
    ].values
    sat_clock_bias1 = epoch_data1.loc[common_sats, "sat_clock_offset_m"].values
    sat_clock_bias2 = epoch_data2.loc[common_sats, "sat_clock_offset_m"].values
    C_obs_m1 = epoch_data1.loc[common_sats, "C_obs_m_corrected"].values
    C_obs_m2 = epoch_data2.loc[common_sats, "C_obs_m_corrected"].values
    L_obs_cycles1 = epoch_data1.loc[common_sats, "L_obs_m_corrected"].values
    L_obs_cycles2 = epoch_data2.loc[common_sats, "L_obs_m_corrected"].values
    sat_ele_rad1 = np.deg2rad(epoch_data1.loc[common_sats, "sat_elevation_deg"].values)
    sat_ele_rad2 = np.deg2rad(epoch_data2.loc[common_sats, "sat_elevation_deg"].values)

    # reorder sat_ele_rad
    sat_ele_rad = np.empty((2 * len(common_sats), 1))
    sat_ele_rad[0::2] = sat_ele_rad1.reshape(-1, 1)
    sat_ele_rad[1::2] = sat_ele_rad2.reshape(-1, 1)

    # reorder code obs
    code_obs = np.empty((2 * len(common_sats), 1))
    code_obs[0::2] = C_obs_m1.reshape(-1, 1)  # even row epoch1
    code_obs[1::2] = C_obs_m2.reshape(-1, 1)  # odd row epoch2

    # reorder cycle obs
    cycle_obs = np.empty((2 * len(common_sats), 1))
    cycle_obs[0::2] = L_obs_cycles1.reshape(-1, 1)
    cycle_obs[1::2] = L_obs_cycles2.reshape(-1, 1)

    # upper part is Code Obs from common sats for both epoch, and low part the Cycle Obs
    y = np.concatenate([code_obs, cycle_obs], axis=0)

    # build D matrix in papper
    num_sats = sat_coor1.shape[0]  # number of satellites
    D = np.zeros((num_sats, 2 * num_sats), dtype=int)
    for i in range(num_sats):
        D[i, 2 * i] = -1
        D[i, 2 * i + 1] = 1

    # build A matrix
    A = block_diag(np.eye(2 * num_sats), D)

    # build TDCP model with A matrix
    def h_A(x_u, x_s1, x_s2):
        hx, jacobian = h_GNSS_code_TDCP(x_u=x_u, x_s1=x_s1, x_s2=x_s2)
        return (A @ hx, A @ jacobian)

    return (
        common_sats,
        sat_coor1,
        sat_coor2,
        sat_clock_bias1,
        sat_clock_bias2,
        y,
        A,
        h_A,
        sat_ele_rad,
    )
