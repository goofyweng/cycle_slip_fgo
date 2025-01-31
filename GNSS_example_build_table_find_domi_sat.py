import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares
from scipy.linalg import block_diag
from ecef2lla import ecef2lla
import scipy.stats as stats
from plot_dist import plot_non_central_chi2, draw_vertical_line
from plot_skyplot import plot_skyplot
from GNSS_code_TDCP_model import build_h_A
from correct_prx_data import correct_prx_code, correct_prx_phase


if __name__ == "__main__":

    # load data
    filepath_csv = "results\TLSE00FRA_R_20240010000_15M_01S_MO.csv"

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

    # print the number of observations
    print(f"There are {len(data_gps_c1c)} GPS L1 C/A observations")

    # correct pseudorange and carrier phase
    code_corrected = correct_prx_code(data_gps_c1c)
    phase_corrected = correct_prx_phase(data_gps_c1c)
    data_gps_c1c["C_obs_m_corrected"] = code_corrected
    data_gps_c1c["L_obs_m_corrected"] = phase_corrected

    # count numbers of detected cycle slips
    print(f"numbers of detected cycle slips:  {(data_gps_c1c.LLI == 1).sum()}")

    # calculate the number of epoch
    epoch = data_gps_c1c["time_of_reception_in_receiver_time"].unique()
    sorted_epochs = sorted(epoch)
    chosen_epochs = [sorted_epochs[2], sorted_epochs[3]]
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

    # number of common satellites, k
    k = len(common_sats)
    # Add np.nan to the first position
    common_sats.insert(0, np.nan)

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

    # create DataFrame to save results
    domi_sat_df = pd.DataFrame(
        columns=["PRN", "z", "chi_pdf_value", "ecef_pos_diff_norm", "fault_pred"]
    )
    # start the idx from carrier phase of first prn at epoch 2
    idx_add_fault = 2 * k + 1
    # manually added fault value
    added_fault_value = 3

    for prn in common_sats:
        x0 = np.random.randn(8, 1)  # normally distributed random init x0 around 0
        # reinitialize the observation to store manually added fault in each iteration
        # make sure we remove the added fault after each iteration and only have one manually added fault
        y_ = np.copy(y)  # copy the non-faulty observation y to y_

        # manually add fault on one satellite carrier phase measurement in y_
        if not np.isnan(prn):  # if prn is not np.nan
            # manually add cycle slip
            y_[idx_add_fault, 0] += added_fault_value
            idx_add_fault += 2  # shift idx to next prn carrier phase
        # check we only have one fault in the faulty observation vector
        check_y = y_ - y
        print(check_y.T)

        # LS
        print("LS start...")
        estimate_result = nonlinear_least_squares(
            h_A, A @ y_, A @ R @ A.T, x0, x_s1=x_s1, x_s2=x_s2
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
        residual_vec = A @ y_ - h_A(estimate_result, x_s1=x_s1, x_s2=x_s2)[0]
        # calculate residual weighted norm, i.e. test statistic z, shape (1x1)
        z = residual_vec.T @ np.linalg.inv(A @ R @ A.T) @ residual_vec
        if np.isnan(prn):
            # keep the ecef position in both epoch as reference
            ref_position = estimate_result[[0, 1, 2, 4, 5, 6]]
            # for no fault estimated position = reference position
            est_position = ref_position
        else:
            est_position = estimate_result[[0, 1, 2, 4, 5, 6]]

        # calculate the pdf value
        n = x0.shape[0]  # number of states
        m = residual_vec.shape[0]  # number of measurements
        chi_pdf_value = stats.ncx2.pdf(z, m - n, 0).item()
        P_fa_set = 0.1  # desired probability of false alarm
        T = stats.chi2.ppf(1 - P_fa_set, m - n)  # threshold

        # compare z with threshold
        if z > T:
            fault_pred = 1  # we predict there is fault, i.e. cycle slip
        else:
            fault_pred = 0  # we predict there is no fault, i.e. no cycle slip

        # calculate the norm difference between estimated position and ref position
        pos_diff_norm = np.linalg.norm(est_position - ref_position)

        # save result into DataFrame
        domi_sat_df.loc[len(domi_sat_df)] = [
            prn,
            z.item(),
            chi_pdf_value,
            pos_diff_norm,
            fault_pred,
        ]

    print(
        f"Manually add fault of {added_fault_value} meters into data...\n{domi_sat_df}"
    )

    # plot skyplot
    satellite_prns = np.array(common_sats[1:])
    user_pos = est_position[0:3].flatten()
    plot_skyplot(
        satellite_positions=sat_coor1,
        user_position=user_pos,
        satellite_prns=satellite_prns,
        epoch=chosen_epochs[0],
    )
    plt.show()
