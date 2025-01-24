import numpy as np
import pandas as pd
from scipy.linalg import block_diag
from GNSS_code_TDCP_model import h_GNSS_code_TDCP
from nonlinear_least_squares import nonlinear_least_squares
from ecef2lla import ecef2lla

# parameters to change phase measurement from cycle to meters
c = 299792458  # speed of light in m/s
fL1 = 1575.42e6  # GPA L1 carrier frequency


def correct_prx_code(df: pd.DataFrame):
    corrected_code_obs = (
        df.C_obs_m
        # + df.sat_clock_offset_m
        + df.relativistic_clock_effect_m
        - df.sagnac_effect_m
        - df.iono_delay_m
        - df.tropo_delay_m
        - df.sat_code_bias_m
    )
    return corrected_code_obs


def correct_prx_phase(df: pd.DataFrame):
    lambda_L1 = c / fL1  # GPS L1 carrier wave length
    corrected_phase_obs = (
        df.L_obs_cycles * lambda_L1
        # + df.sat_clock_offset_m
        + df.relativistic_clock_effect_m
        - df.sagnac_effect_m
        + df.iono_delay_m
        - df.tropo_delay_m
    )
    return corrected_phase_obs


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
    C_obs_m1 = epoch_data1.loc[common_sats, "C_obs_m_corrected"].values
    C_obs_m2 = epoch_data2.loc[common_sats, "C_obs_m_corrected"].values
    L_obs_cycles1 = epoch_data1.loc[common_sats, "L_obs_m_corrected"].values
    L_obs_cycles2 = epoch_data2.loc[common_sats, "L_obs_m_corrected"].values

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

    # load data, ref Latitude, Longitude: 43.560694, 1.480872
    # refer from: https://network.igs.org/TLSE00FRA
    # filepath_csv = "TLSE00FRA_R_20240010100_15M_30S_MO.csv"
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

    # correct pseudorange and carrier phase
    code_corrected = correct_prx_code(data_gps_c1c)
    phase_corrected = correct_prx_phase(data_gps_c1c)
    data_gps_c1c["C_obs_m_corrected"] = code_corrected
    data_gps_c1c["L_obs_m_corrected"] = phase_corrected

    # calculate the number of epoch
    epoch = data_gps_c1c["time_of_reception_in_receiver_time"].unique()
    sorted_epochs = sorted(epoch)
    chosen_epochs = [sorted_epochs[0], sorted_epochs[1]]
    print(f"Selected epochs {chosen_epochs}")

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
    ) = build_h_A(data_gps_c1c, chosen_epochs)

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
    print(f"Estimated ECEF at epoch 1: {estimate_result[0:3].flatten()}")
    print(f"Estimated ECEF at epoch 2: {estimate_result[4:].flatten()}")
    print(f"Estimated LLA at epoch 1: {estimate_lla_epoch1}")
    print(f"Estimated LLA at epoch 2: {estimate_lla_epoch2}")
    print("LS done...")
    
    print('done...')
