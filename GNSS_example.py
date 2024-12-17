import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from nonlinear_least_squares import nonlinear_least_squares


def build_h(data, epoch):

    # Choose the data from the same epoch
    epoch_data = data[data["time_of_reception_in_receiver_time"] == epoch]
    
    if epoch_data.empty:
        print(f"No data found for epoch {epoch}")
        return None

    # Etract data from this epoch
    visible_sat = epoch_data["prn"].tolist()
    sat_coor = epoch_data[["sat_pos_x_m", "sat_pos_y_m", "sat_pos_z_m"]].values
    sat_clock_bias = epoch_data["sat_clock_offset_m"].values
    c_obs_m = epoch_data["C_obs_m"].values

    # model for toy GNSS example, nonlinear
    def h_toy_GNSS_w_usr_clk_b(x_u, x_s):
        r = np.linalg.norm(x_s[0:3,:] - x_u[0:3], axis=0).reshape(-1, 1)  # geometric range, (mx1)
        rho = r + (x_u[-1] - x_s[-1,:]).reshape(-1,1) # add clock bias to build pseudorange
        jacobian = np.hstack((-1 * (x_s[0:3,:] - x_u[0:3]).T / r, np.ones((rho.size,1))))  # each row is a unit vector from user to sat
        return (rho, jacobian)
    

    return visible_sat, sat_coor, sat_clock_bias, c_obs_m, h_toy_GNSS_w_usr_clk_b


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
    first_epoch = sorted_epochs[0]
    visible_sat, sat_coor, sat_clock_bias, c_obs_m, h = build_h(data_gps_c1c, first_epoch)
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

