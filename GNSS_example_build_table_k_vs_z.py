"""
This script produce a table to show the relation between 
different number of satellites, k, and the corresponding 
weighted residual norm, z, using real-world GNSS data.
The table is shown in the termianl at the end.
Several figures will be generated to show the centraled 
chi-squared pdf with DOF = m-n, where m is the # of measurements
and n is the number of user states to be estimated. 
The calculated Threshold, T, and z are also shown in those figures.

The used GNSS data is checked if there is a cycle slip 
with the 'filter_chosen_epochs()' function.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares
from scipy.linalg import block_diag
from ecef2lla import ecef2lla
import scipy.stats as stats
from plot_dist import plot_non_central_chi2, draw_vertical_line
from GNSS_code_TDCP_model import h_GNSS_code_TDCP
from filter_epoch_fnc import create_LLI_label_list
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import glob
import os


def filter_chosen_epochs_no_fault(chosen_epoch_array):
    """
    The function filter the chosen_epoch_array and keep the row which
    satisfies one of the following conditions:
    1. 'No Fault' in epoch t, and 'No Fault' in epoch t+1
    =====
    Input: chosen_epoch_array with the columns defined as follows:
    1st column: epoch t
    2nd column: epoch t+1
    3rd and 4th column: fault flag, can be "no", "single", "multiple" or
    "no fault if exclude SAT {prn_fault_epoch1} in previous epoch"
    =====
    Output: chosen_epoch_array_filtered filtered by the conditions.
    """
    # Filter rows based on the condition
    condition1 = (chosen_epoch_array[:, 2] == "no") & (chosen_epoch_array[:, 3] == "no")

    # Retrieve the rows
    chosen_epoch_array_filtered = chosen_epoch_array[condition1]

    return chosen_epoch_array_filtered


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
    # Directory containing the CSV files
    csv_directory = "results"
    # Use glob to find all CSV files matching the pattern
    filepath_csv = glob.glob(
        os.path.join(csv_directory, "TLSE00FRA_R_*_15M_01S_MO.csv")
    )

    # Initialize an empty list to hold individual DataFrames
    dataframes = []
    # Loop through each file and read it
    for file in filepath_csv[:4]:
        # parse cv and create pd.DataFrame
        data_prx = pd.read_csv(
            file,
            comment="#",  # ignore lines beginning with '#'
            parse_dates=[
                "time_of_reception_in_receiver_time"
            ],  # consider columns "time_of_reception_in_receiver_time"  as pd.Timestamp
        )

        # filter the data
        data_gps_c1c = data_prx[
            (data_prx["constellation"] == "G")  # keep only GPS data
            & (
                data_prx["rnx_obs_identifier"] == "1C"
            )  # keep only L1C/A observations (this comes from the RINEX format)
        ].reset_index(
            drop=True
        )  # reset index of the DataFrame in order to have a continuous range of integers, after deleting some lines

        # Append the DataFrame to the list
        dataframes.append(data_gps_c1c)

    data_gps_c1c = pd.concat(dataframes, ignore_index=True)

    # drop unclean data
    data_gps_c1c = data_gps_c1c.dropna()
    # print the number of observations
    print(f"There are {len(data_gps_c1c)} GPS L1 C/A observations")

    # chose the epochs we want
    chosen_epochs = create_LLI_label_list(data_gps_c1c)
    chosen_epochs_filtered = filter_chosen_epochs_no_fault(chosen_epochs)
    # chose the first pair of chosen_epochs_filtered
    chosen_epoch_first_pair = chosen_epochs_filtered[0]
    print(f"Selected epochs {chosen_epoch_first_pair}")

    # find the all common satellites in the chosen epoch
    (common_sats_all, _, _, _, _, _, _, _) = build_h_A(data_gps_c1c, chosen_epoch_first_pair)

    # build the num_sat vector for different '# common satellites' or 'k'
    # ex: when there are 8 common_sats, k_vec = [4, 5, 6, 7, 8]
    # Note that we still need 4 satellites to estimate the user states...
    # Because the last k rows of A@h is linearly dependent to the first 2k rows of h,
    # where h is the jacobian matrix 
    k_vec = np.arange(4, len(common_sats_all) + 1) 
    # store the calculated residual weighted norm
    z_vec = np.zeros([k_vec.shape[0], ])

    idx = 0 # idx used to save z result
    # loop throught different k
    for k in k_vec:
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
        ) = build_h_A_num_sats(data_gps_c1c, chosen_epoch_first_pair, k)

        # covariance matrix
        sigma_code = 3
        sigma_phase = 0.019
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
        # convert latitude and longtitude from radiance to degree
        estimate_lla_epoch1[:2] *= 180 / np.pi
        estimate_lla_epoch2[:2] *= 180 / np.pi
        print(f"Estimated LLA at epoch 1: {estimate_lla_epoch1}")
        print(f"Estimated LLA at epoch 2: {estimate_lla_epoch2}")
        print("LS done...")

        # calculate residual vector, shape (3kx1)
        residual_vec = A @ y - h_A(estimate_result, x_s1=x_s1, x_s2=x_s2)[0]
        # calculate residual weighted norm, i.e. test statistic z, shape (1x1)
        z = residual_vec.T @ np.linalg.inv(A @ R @ A.T) @ residual_vec

        # save z result
        z_vec[idx] = z.item()

        # fault detection
        # define the threshold T, by the probability of false alarm
        n = x0.shape[0]  # number of states
        m = residual_vec.shape[0]  # number of measurements
        P_fa_set = 0.1  # desired probability of false alarm
        T = stats.chi2.ppf(1 - P_fa_set, m - n)  # threshold

        # plot z and T on the chi-squared pdf
        fig, ax = plt.subplots()
        plot_non_central_chi2(ax, m - n, 0, xlim=100)
        draw_vertical_line(ax, T, "red", f"T={T:.4f}")
        draw_vertical_line(ax, z, "blue", f"z={z.item():.4f}")
        ax.legend()
        ax.set_title(f"number of satellites, k={k}")
    
        # idx increase
        idx += 1

    plt.show()
    # Create a pandas DataFrame
    df = pd.DataFrame({
        '# of satellites (k)': k_vec,  
        'weighted residual norm (z)': z_vec  
    })
    print(df)


