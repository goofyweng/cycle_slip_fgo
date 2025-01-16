import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares
from scipy.linalg import block_diag
from ecef2lla import ecef2lla
import scipy.stats as stats
from plot_dist import plot_non_central_chi2, draw_vertical_line

import numpy as np

import numpy as np

import numpy as np

def epochs_with_LLI_GPS(data_gps_c1c):

    # Extract and sort unique epochs
    epoch_list = np.sort(data_gps_c1c["time_of_reception_in_receiver_time"].unique())
    
    # Find epochs with LLI sum == 1
    epochs_with_LLI1 = (
        data_gps_c1c[data_gps_c1c["LLI"] > 0]
        .groupby("time_of_reception_in_receiver_time")
        .filter(lambda group: group["LLI"].sum() == 1)
        ["time_of_reception_in_receiver_time"]
        .unique()
    )
    
    # Find epochs with LLI sum == 0
    epochs_with_LLI0 = (
        data_gps_c1c
        .groupby("time_of_reception_in_receiver_time")
        .filter(lambda group: group["LLI"].sum() == 0)
        ["time_of_reception_in_receiver_time"]
        .unique()
    )
    
    # Initialize chosen_epoch array
    chosen_epoch = []
    
    # Iterate over epochs_with_LLI1
    for epoch in epochs_with_LLI1:
        # Ensure epoch and pre_epoch are in numpy.datetime64 format
        epoch = np.datetime64(epoch)  # Make sure it's in the correct format
        
        # Find index of the current epoch
        ind_epoch = np.where(epoch_list == epoch)[0][0]
        
        # Get the previous epoch if it exists
        if ind_epoch > 0:
            pre_epoch = epoch_list[ind_epoch - 1]
            pre_epoch = np.datetime64(pre_epoch)  # Make sure it's in the correct format
            
            # Determine which list the previous epoch belongs to
            if pre_epoch in epochs_with_LLI0:
                chosen_epoch.append([pre_epoch, epoch, "LLI1, LLI0"])
            elif pre_epoch in epochs_with_LLI1:
                chosen_epoch.append([pre_epoch, epoch, "LLI1, LLI1"])
            else:
                chosen_epoch.append([None, epoch, "Others"])
        else:
            chosen_epoch.append([None, epoch, "No Previous Epoch"])
    
    # Convert the chosen_epoch list to a NumPy array or DataFrame for further use
    chosen_epoch_array = np.array(chosen_epoch, dtype=object)
    
    return chosen_epoch_array  # [epoch, pre_epoch, flag]



def build_h_A(data, epoch):

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
    C_obs_m1 = epoch_data1.loc[common_sats, "C_obs_m"].values
    C_obs_m2 = epoch_data2.loc[common_sats, "C_obs_m"].values
    L_obs_cycles1 = epoch_data1.loc[common_sats, "L_obs_cycles"].values
    L_obs_cycles2 = epoch_data2.loc[common_sats, "L_obs_cycles"].values
    # phase measurement from cycle to meters
    c = 299792458
    fL1 = 1575.42e6
    lambda_L1 = c / fL1
    L_obs_cycles1 = L_obs_cycles1 * lambda_L1
    L_obs_cycles2 = L_obs_cycles2 * lambda_L1

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
        h_x = np.vstack((rho, phase)) # shape should be (4kx1)

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
        jacobian = np.vstack((jacobian_rho, jacobian_rho)) # shape should be (4kx8)

        return (h_x, jacobian)

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
    )


if __name__ == "__main__":

    # load data
    filepath_csv = "TLSE00FRA_R_20240010100_15M_30S_MO.csv"

    # parse cv and create pd.DataFrame
    data_prx = pd.read_csv(
        filepath_csv,
        comment="#",  # ignore lines beginning with '#'
        parse_dates=[
            "time_of_reception_in_receiver_time"
        ],  # consider columns "time_of_reception_in_receiver_time"  as pd.Timestamp
    )

    # filter the data
    data_gps_c1c = data_prx[
        (data_prx["constellation"] == "G")  # keep only GPS data
        & (data_prx["rnx_obs_identifier"] == "1C")  # keep only L1C/A observations (this comes from the RINEX format)
    ].reset_index(
        drop=True
    )  # reset index of the DataFrame in order to have a continuous range of integers, after deleting some lines

    # print the number of observations
    print(f"There are {len(data_gps_c1c)} GPS L1 C/A observations")

    # display first rows of DataFrame
    print(data_gps_c1c.head())

    # display existing columns
    print(data_gps_c1c.columns)

    # count numbers of detected cycle slips
    print(f"numbers of detected cycle slips:  {(data_gps_c1c.LLI == 1).sum()}")

    # calculate the number of epoch
    chosen_epochs = epochs_with_LLI_GPS(data_gps_c1c)
    # print(f"Selected epochs {chosen_epochs}")

    fault_pre_vec = np.zeros([chosen_epochs.shape[0], 1])
    z_vec = np.zeros([chosen_epochs.shape[0], 1])
    idx = 0
    for chosen_epoch in chosen_epochs:
        if chosen_epoch[0] is None:
            fault_pre_vec[idx] = np.nan
            z_vec[idx] = np.nan
            break
        else:
            # print(f"Selected epochs {chosen_epoch[0:2]}")
            # check if we have LLI == 1, i.e. cycle slip, in the selected epochs
            LLI_chosen_epochs = data_gps_c1c[
                (data_gps_c1c["time_of_reception_in_receiver_time"].isin(chosen_epoch))
            ].LLI.to_numpy()
            fault_true = np.any(LLI_chosen_epochs)
            print(f"Cycle clip in selected epochs? => {fault_true}")

            # get the satellite coordinates and clock bias at 1st and 2nd epochs

            (
                common_sats,  # satellite prns exist in both 1st and 2nd epoch
                sat_coor1,  # satellite coordinate at 1st epoch
                sat_coor2,  # satellite coordinate at 2nd epoch
                sat_clock_bias1,  # satellite clock bias at 1st epoch
                sat_clock_bias2,  # satellite clock bias at 2nd epoch
                y,  # obseravtion vector
                A,  # A matrix used to construch TDCP
                h_A,  # (A @ h) function
            ) = build_h_A(data_gps_c1c, chosen_epoch)

            # covariance matrix
            sigma_code = 3
            sigma_phase = 1
            # factor uncerntainty model, R has shape (4kx4k), k is # of sats appears in both 1st and 2nd epoch
            R = block_diag(
                sigma_code**2 * np.eye(2 * sat_coor1.shape[0]),
                sigma_phase**2 * np.eye(2 * sat_coor1.shape[0]),
            )

            # satellite states at first epoch
            x_s1 = np.vstack((sat_coor1.T, sat_clock_bias1))
            # satellite states at second epoch
            x_s2 = np.vstack((sat_coor2.T, sat_clock_bias2))

            np.random.seed(42)  # Set a seed for reproducibility
            x0 = np.random.randn(8, 1)  # normally distributed random init x0 around 0
            # LS
            print("LS start...")
            estimate_result = nonlinear_least_squares(
                h_A, A @ y, A @ R @ A.T, x0, x_s1=x_s1, x_s2=x_s2
            )
            # convert estimate position from ECEF to LLA
            estimate_lla_epoch1 = np.array(
                ecef2lla(estimate_result[0], estimate_result[1], estimate_result[2])
            ).reshape(-1)
            estimate_lla_epoch2 = np.array(
                ecef2lla(estimate_result[4], estimate_result[5], estimate_result[6])
            ).reshape(-1)
            estimate_lla_epoch1[:2] *= 180 / np.pi
            estimate_lla_epoch2[:2] *= 180 / np.pi
            print(f"Estimated LLA at epoch 1: {estimate_lla_epoch1}")
            print(f"Estimated LLA at epoch 2: {estimate_lla_epoch2}")
            print("LS done...")

            # calculate residual vector, shape (3kx1)
            residual_vec = A @ y - h_A(estimate_result, x_s1=x_s1, x_s2=x_s2)[0]
            # calculate residual weighted norm, i.e. test statistic z, shape (1x1)
            z = residual_vec.T @ np.linalg.inv(A @ R @ A.T) @ residual_vec

            # fault detection
            # define the threshold T, by the probability of false alarm
            n = x0.shape[0]  # number of states
            m = residual_vec.shape[0]  # number of measurements
            P_fa_set = 0.1  # desired probability of false alarm
            T = stats.chi2.ppf(1 - P_fa_set, m - n)  # threshold

            # compare z with threshold
            if z > T:
                fault_pred = 1  # we predict there is fault, i.e. cycle slip
                fault_pre_vec[idx] = fault_pred
                z_vec[idx] = z
            else:
                fault_pred = 0  # we predict there is no fault, i.e. no cycle slip
                fault_pre_vec[idx] = fault_pred
                z_vec[idx] = z
            print(f"Cycle slip detected? => {bool(fault_pred)}")

            # index increament for each iteration
            idx += 1

            # # plot z and T on the chi-squared pdf
            # fig, ax = plt.subplots()
            # plot_non_central_chi2(ax, m - n, 0, xlim=60)
            # draw_vertical_line(ax, T, "red", "T")
            # draw_vertical_line(ax, z, "blue", "z")
            # ax.legend()


        
    # plt.show()

    fault_pred_result = np.hstack([chosen_epochs, fault_pre_vec, z_vec])
    print(fault_pred_result[:,[0,1,-2,-1]])
    