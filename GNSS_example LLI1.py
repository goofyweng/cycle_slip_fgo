import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares
from scipy.linalg import block_diag
from ecef2lla import ecef2lla
import scipy.stats as stats
from plot_dist import plot_non_central_chi2, draw_vertical_line
from GNSS_code_TDCP_model import build_h_A
from filter_epoch_fnc import create_LLI_label_list, select_evenly_dist_true
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from correct_prx_data import correct_prx_code, correct_prx_phase
import glob
import os


def filter_chosen_epochs(chosen_epoch_array):
    """
    The function filter the chosen_epoch_array and keep the row which
    satisfies one of the following three conditions:
    1. 'No Fault' in epoch t, and 'No Fault' in epoch t+1
    2. 'No Fault' in epoch t, and 'Single Fault' in epoch t+1
    3. 'Single Fault' in epoch t, and 'Single Fault' in epoch t+1
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
    condition2 = (chosen_epoch_array[:, 2] == "no") & (
        chosen_epoch_array[:, 3] == "single"
    )
    condition3 = (chosen_epoch_array[:, 2] == "single") & (
        chosen_epoch_array[:, 3] == "single"
    )

    # calculate number of chosen_epochs fulfill condition 2 and 3
    num_cond2_cond3 = np.sum(condition2) + np.sum(condition3)
    if num_cond2_cond3 >= 5:  # if there are more than 5 'true no fault' events
        # within condition1, evenly select and keep the same number of true elements as the sum
        # of the number of true elements in condition2 and condition3
        condition1 = select_evenly_dist_true(
            bool_array=condition1, num_true=num_cond2_cond3
        )
    # Combined conditions, union condition of condition1 and condition2
    final_condition = condition1 | condition2 | condition3
    # Retrieve the rows
    chosen_epoch_array_filtered = chosen_epoch_array[final_condition]

    return chosen_epoch_array_filtered


if __name__ == "__main__":

    # load data, ref Latitude, Longitude: 43.560694, 1.480872
    # refer from: https://network.igs.org/TLSE00FRA
    # Directory containing the CSV files
    csv_directory = "results"
    # filepath_csv = "TLSE00FRA_R_20240010100_15M_30S_MO.csv"
    # filepath_csv = ["TLSE00FRA_R_20240010000_15M_01S_MO.csv",
    #                 "TLSE00FRA_R_20240010015_15M_01S_MO.csv"]
    # Use glob to find all CSV files matching the pattern
    filepath_csv = glob.glob(
        os.path.join(csv_directory, "TLSE00FRA_R_*_15M_01S_MO.csv")
    )

    # Initialize an empty list to hold individual DataFrames
    dataframes = []
    # number of files to be processed, set between 1 to len(filepath_csv)=96,
    # the duration of one file is 15 minutes
    num_files = 4
    # Loop through each file and read it
    for file in filepath_csv[:num_files]:
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
    chosen_epochs_filtered = filter_chosen_epochs(chosen_epochs)

    # save predicted fault results
    fault_pre_vec = np.zeros([chosen_epochs_filtered.shape[0], 1])
    # save true fault results
    fault_true_vec = np.zeros([chosen_epochs_filtered.shape[0], 1])

    # save T and z results
    T_vec = np.zeros([chosen_epochs_filtered.shape[0], 1])
    z_vec = np.zeros([chosen_epochs_filtered.shape[0], 1])
    idx = 0

    for chosen_epoch in chosen_epochs_filtered:
        if chosen_epoch[0] is None:
            fault_pre_vec[idx] = np.nan
            fault_true_vec[idx] = np.nan
            z_vec[idx] = np.nan
            break
        else:
            print(f"Selected epochs {chosen_epoch[0:2]}")
            # check if we have LLI == 1, i.e. cycle slip, in the selected epochs
            LLI_chosen_epochs = data_gps_c1c[
                (data_gps_c1c["time_of_reception_in_receiver_time"].isin(chosen_epoch))
            ].LLI.to_numpy()
            fault_true = np.any(LLI_chosen_epochs)
            print(f"Cycle clip in selected epochs? => {fault_true}")

            # save true fault vector
            if chosen_epoch[2] == "single" and chosen_epoch[3] == "single":
                fault_true_vec[idx] = 1  # true fault
            elif chosen_epoch[2] == "no" and chosen_epoch[3] == "single":
                fault_true_vec[idx] = 1  # true fault
            else:
                fault_true_vec[idx] = 0  # true no fault

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
            ) = build_h_A(data_gps_c1c, chosen_epoch)

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
            P_fa_set = 0.05  # desired probability of false alarm
            T = stats.chi2.ppf(1 - P_fa_set, m - n)  # threshold

            # compare z with threshold
            if z > T:
                fault_pred = 1  # we predict there is fault, i.e. cycle slip
                fault_pre_vec[idx] = fault_pred
                z_vec[idx] = z
                T_vec[idx] = T
            else:
                fault_pred = 0  # we predict there is no fault, i.e. no cycle slip
                fault_pre_vec[idx] = fault_pred
                z_vec[idx] = z
                T_vec[idx] = T
            print(f"Cycle slip detected? => {bool(fault_pred)}")

            # index increament for each iteration
            idx += 1

    # horizontally stack useful vectors
    fault_pred_result = np.hstack(
        [chosen_epochs_filtered, fault_true_vec, fault_pre_vec, z_vec, T_vec]
    )
    # save prediction result into df
    fault_pred_result_df = pd.DataFrame(
        data=fault_pred_result,
        columns=[
            "chosen_epoch_t",
            "chosen_epoch_t+1",
            "LLI_epoch_t",
            "LLI_epoch_t+1",
            "fault_true",
            "fault_pre",
            "z",
            "T",
        ],
    )

    # build confusion matrix
    cm = confusion_matrix(fault_true_vec, fault_pre_vec)
    print(cm)
    # calculate emperical Probability of false alarm
    TN = cm[0, 0]  # True Negative
    FP = cm[0, 1]  # False Positive
    FN = cm[1, 0]  # False Negative
    TP = cm[1, 1]  # True Positive
    P_fa = FP / (FP + TN)
    print(f"Emperical P_fa={P_fa:.4f}, P_fa_set={P_fa_set}")
    # make confusion matrix values into percentage
    cm = cm / np.sum(cm) * 100
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    disp.ax_.set_xticklabels(["No fault", "Fault"])
    disp.ax_.set_yticklabels(["No fault", "Fault"])
    disp.ax_.set_title(f"Emperical P_fa={P_fa:.4f}, P_fa_set={P_fa_set}")
    plt.show()
