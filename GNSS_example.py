"""
This script processes GPS data to analyze satellite indicators.
It calculates TDCP for common satellites in two consecutive epochs
and verified the LLI values that given by the data.
 
The script:
1. Loads the GPS data
2. Choose two epochs with LLI = 0
3. Calculates it's square norm residual and plot the distribution

Created on: 2024-01
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares
from scipy.linalg import block_diag
from ecef2lla import ecef2lla
import scipy.stats as stats
from plot_dist import plot_non_central_chi2, draw_vertical_line
from GNSS_code_TDCP_model import build_h_A
from correct_prx_data import correct_prx_code, correct_prx_phase


if __name__ == "__main__":

    # load data
    filepath_csv = "TLSE00FRA_R_20240010000_15M_01S_MO.csv"

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
        & (
            data_prx["rnx_obs_identifier"] == "1C"
        )  # keep only L1C/A observations (this comes from the RINEX format)
    ].reset_index(
        drop=True
    )  # reset index of the DataFrame in order to have a continuous range of integers, after deleting some lines

    # drop unclean data
    data_gps_c1c = data_gps_c1c.dropna()

    # correct pseudorange and carrier phase
    code_corrected = correct_prx_code(data_gps_c1c)
    phase_corrected = correct_prx_phase(data_gps_c1c)
    data_gps_c1c["C_obs_m_corrected"] = code_corrected
    data_gps_c1c["L_obs_m_corrected"] = phase_corrected

    # print the number of observations
    print(f"There are {len(data_gps_c1c)} GPS L1 C/A observations")

    # count numbers of detected cycle slips
    print(f"numbers of detected cycle slips:  {(data_gps_c1c.LLI == 1).sum()}")

    # calculate the number of epoch
    epoch = data_gps_c1c["time_of_reception_in_receiver_time"].unique()
    sorted_epochs = sorted(epoch)
    chosen_epochs = [sorted_epochs[0], sorted_epochs[1]]
    print(f"Selected epochs {chosen_epochs}")

    # check if we have LLI == 1, i.e. cycle slip, in the selected epochs
    LLI_chosen_epochs = data_gps_c1c[
        (data_gps_c1c["time_of_reception_in_receiver_time"].isin(chosen_epochs))
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
        sat_ele_rad,
    ) = build_h_A(data_gps_c1c, chosen_epochs)

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
    else:
        fault_pred = 0  # we predict there is no fault, i.e. no cycle slip
    print(f"Cycle slip detected? => {bool(fault_pred)}")

    # plot z and T on the chi-squared pdf
    fig, ax = plt.subplots()
    plot_non_central_chi2(ax, m - n, 0, xlim=60)
    draw_vertical_line(ax, T, "red", "T")
    draw_vertical_line(ax, z, "blue", "z")
    ax.legend()
    plt.show()
