import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares


def build_h(data, epoch):

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

    sat_coor1 = epoch_data1.loc[common_sats, ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].values
    sat_coor2 = epoch_data2.loc[common_sats, ["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].values
    sat_clock_bias1 = epoch_data1.loc[common_sats, "sat_clock_offset_m"].values
    sat_clock_bias2 = epoch_data2.loc[common_sats, "sat_clock_offset_m"].values
    C_obs_m1 = epoch_data1.loc[common_sats, "C_obs_m"].values
    C_obs_m2 = epoch_data2.loc[common_sats, "C_obs_m"].values
    L_obs_cycles1 = epoch_data1.loc[common_sats, "L_obs_cycles"].values
    L_obs_cycles2 = epoch_data2.loc[common_sats, "L_obs_cycles"].values

    # reorder code obs
    code_obs = np.empty((2 * len(common_sats), 1)) 
    code_obs[0::2] = C_obs_m1.reshape(-1, 1)  # even row epoch1
    code_obs[1::2] = C_obs_m2.reshape(-1, 1)  # odd row epoch2

    # reorder cycle obs
    cycle_obs = np.empty((2 * len(common_sats), 1))  
    cycle_obs[1::2] = L_obs_cycles2.reshape(-1, 1)

    # upper part is Code Obs from common sats for both epoch, and low part the Cycle Obs
    h = np.concatenate([code_obs, cycle_obs], axis=0)

    # model for toy GNSS example, nonlinear
    def h_toy_GNSS_w_usr_clk_b(x_u, x_s):
        r = np.linalg.norm(x_s[0:3,:] - x_u[0:3], axis=0).reshape(-1, 1)  # geometric range, (mx1)
        rho = r + (x_u[-1] - x_s[-1,:]).reshape(-1,1) # add clock bias to build pseudorange
        jacobian = np.hstack((-1 * (x_s[0:3,:] - x_u[0:3]).T / r, np.ones((rho.size,1))))  # each row is a unit vector from user to sat
        return (rho, jacobian)

    return common_sats, sat_coor1, sat_coor2, sat_clock_bias1, sat_clock_bias2, \
    C_obs_m1, C_obs_m2, L_obs_cycles1, L_obs_cycles2, h_toy_GNSS_w_usr_clk_b


if __name__ == "__main__":

    # load data
    filepath_csv = "TLSE00FRA_R_20240010100_15M_30S_MO.csv"

    # parse cv and create pd.DataFrame
    data_prx = pd.read_csv(
            filepath_csv,
            comment="#", # ignore lines beginning with '#' 
            parse_dates=["time_of_reception_in_receiver_time"], # consider columns "time_of_reception_in_receiver_time"  as pd.Timestamp
        )

    # filter the data
    data_gps_c1c = data_prx[
        (data_prx["constellation"] == "G")  # keep only GPS data
        & (data_prx["rnx_obs_identifier"] == "1C")  # keep only L1C/A observations (this comes from the RINEX format)
    ].reset_index(drop=True) # reset index of the DataFrame in order to have a continuous range of integers, after deleting some lines

    # print the number of observations
    print(
        f"There are {len(data_gps_c1c)} GPS L1 C/A observations"
    )

    # display first rows of DataFrame
    print(data_gps_c1c.head())

    # display existing columns
    print(data_gps_c1c.columns)

    # plot all carrier phase observations in the same plot
    fig, ax = plt.subplots()
    data_gps_c1c.groupby( 
        "prn"  # creates group according to value of "prn" column
        ).plot(  # calls plot function on each group
            x="time_of_reception_in_receiver_time",
            y="L_obs_cycles",
            ax=ax,  # plot on the same graph
        );
    plt.legend(data_gps_c1c.groupby("prn").groups.keys())

    # count numbers of detected cycle slips
    print(f" numbers of detected cycle slips:  {(data_gps_c1c.LLI == 1).sum()}")


    # calculate the number of epoch
    epoch = data_gps_c1c["time_of_reception_in_receiver_time"].unique() 
    sorted_epochs = sorted(epoch)
    chosen_epochs = [sorted_epochs[0], sorted_epochs[1]]
    visible_sat, sat_coor, sat_clock_bias, c_obs_m, h = build_h(data_gps_c1c, chosen_epochs)
    R = np.eye(c_obs_m.size)

    # satellite states
    x_s = np.vstack((sat_coor.T, sat_clock_bias))
    x0 = np.array([0, 0, 0, 0]).reshape(-1, 1)
    # LS
    estimate_result = nonlinear_least_squares(h, c_obs_m.reshape(-1,1), R, x0, x_s=x_s)


    print("Visible Satellites:", visible_sat)
    print("Satellite Coordinates:", sat_coor)
    print("Satellite Clock Bias:", sat_clock_bias)
    print("Observations (c_obs_m):", c_obs_m)
    plt.show()

