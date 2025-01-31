"""
This script generate the histogram of weighted residual norm, 'z',
using the desired number of satellites, 'k', in the consecutive epochs.
The histogram is overlaped with the theoretical chi-squared PDF of z, 
and the calculated threshold for fault detection, 'T'.
The input prx data is first filtered to ensure we only consider epochs
with LLI==0. 
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares
from scipy.linalg import block_diag
from ecef2lla import ecef2lla
import scipy.stats as stats
from plot_dist import plot_non_central_chi2, draw_vertical_line
from GNSS_code_TDCP_model import build_h_A, build_h_A_num_sats
from filter_epoch_fnc import create_LLI_label_list
from correct_prx_data import correct_prx_code, correct_prx_phase
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
    for file in filepath_csv[:16]:
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

    # correct pseudorange and carrier phase
    code_corrected = correct_prx_code(data_gps_c1c)
    phase_corrected = correct_prx_phase(data_gps_c1c)
    data_gps_c1c["C_obs_m_corrected"] = code_corrected
    data_gps_c1c["L_obs_m_corrected"] = phase_corrected

    # print the number of observations
    print(f"There are {len(data_gps_c1c)} GPS L1 C/A observations")

    # chose the epochs we want
    chosen_epochs = create_LLI_label_list(data_gps_c1c)
    chosen_epochs_filtered = filter_chosen_epochs_no_fault(chosen_epochs)
    # test with shorter chosen_epochs_filtered
    # chosen_epochs_filtered = chosen_epochs_filtered[:100]

    # number of satellites we want to use in the filtered chosen epochs
    max_num_sat = 12
    min_num_sat = 7
    k_vec = np.arange(min_num_sat, max_num_sat+1)  

    # store the calculated residual weighted norm
    z_vec = np.zeros(
        [
            k_vec.shape[0], chosen_epochs_filtered.shape[0]
        ]
    )
    # store the calculated threshold
    T_vec = np.zeros(
        k_vec.shape,
    )
    # store the corresponding DOF of chi-squared dist.
    dof = np.zeros(
        k_vec.shape,
    )
    col_idx_z_vec = 0  # idx used to save z result
    idx = 0 # idx used to save T and dof

    # within all filtered chosen epochs, calculate the weighted residual norm, z,
    # for specific number of satellite we want to use, k
    for k in k_vec:

        for epoch_pair in chosen_epochs_filtered:
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
                sat_ele_rad,
            ) = build_h_A_num_sats(data_gps_c1c, epoch_pair, k)

            # if the common satellites in the current epoch is less than k
            # then continue to the next loop
            if len(common_sats) < k:
                continue

            # covariance matrix
            sigma_code = np.diag(1 / np.sin(sat_ele_rad).flatten())
            sigma_phase = np.diag(0.05 / np.sin(sat_ele_rad).flatten())
            # factor uncerntainty model, R has shape (4kx4k), k is # of sats appears in both 1st and 2nd epoch
            R = block_diag(np.square(sigma_code), np.square(sigma_phase))

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
            print(f"Estimated LLA at epoch {epoch_pair[0]}: {estimate_lla_epoch1}")
            print(f"Estimated LLA at epoch {epoch_pair[1]}: {estimate_lla_epoch2}")
            print("LS done...")

            # calculate residual vector, shape (3kx1)
            residual_vec = A @ y - h_A(estimate_result, x_s1=x_s1, x_s2=x_s2)[0]
            # calculate residual weighted norm, i.e. test statistic z, shape (1x1)
            z = residual_vec.T @ np.linalg.inv(A @ R @ A.T) @ residual_vec


            # save z result
            z_vec[idx, col_idx_z_vec] = z.item()
            # idx increase
            col_idx_z_vec += 1

        # plot theoretical pdf
        n = x0.shape[0]  # number of states
        m = residual_vec.shape[0]  # number of measurements
        P_fa_set = 0.1  # desired probability of false alarm
        T = stats.chi2.ppf(1 - P_fa_set, m - n)  # threshold
        # save result for T and dof
        dof[idx] = m-n
        T_vec[idx] = T

        # increas idx and reset col_idx_z_vec for the calculation using next k
        idx +=1
        col_idx_z_vec= 0


    # Create a figure and axis
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    # Plot results for different k
    for j in range(k_vec.shape[0]):
        row, col = divmod(j, 3)  # Calculate row and column index
        plot_non_central_chi2(ax[row, col], dof[j], 0, xlim=100)
        draw_vertical_line(ax[row, col], T_vec[j], "red", f"T={T_vec[j]:.4f}")
        ax[row, col].hist(
            z_vec[j,:],
            # bins=30,
            density=True,
            alpha=0.5,
            edgecolor="black",
            label=f"$\\text{{z}}$",
        )
        ax[row, col].legend()
        ax[row, col].set_title(f"Number of satellites k={k_vec[j]}")
    
    # Save the result of multiple arrays into a single file
    np.savez("result_data.npz", k_vec=k_vec, z_vec=z_vec, T_vec=T_vec, dof=dof)

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()
    print("done")
