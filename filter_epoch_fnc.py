import pandas as pd
import numpy as np

def create_LLI_label_list(data_gps_c1c):

    epoch_list = data_gps_c1c["time_of_reception_in_receiver_time"].unique()
    # Filter rows with LLI > 0 (indicating cycle slips)
    data_gps_c1c_LLI = data_gps_c1c[data_gps_c1c["LLI"] > 0].reset_index(drop=True)

    # Group by epoch and check LLI == 0 
    epochs_with_LLI0 = (
        data_gps_c1c
        .groupby("time_of_reception_in_receiver_time")
        .filter(lambda group: group["LLI"].sum() == 0)
        ["time_of_reception_in_receiver_time"]
        .unique() 
    )

    # Group by epoch and check LLI == 1
    epochs_with_LLI1 = (
        data_gps_c1c
        .groupby("time_of_reception_in_receiver_time")
        .filter(lambda group: group["LLI"].sum() == 1)
        ["time_of_reception_in_receiver_time"]
        .unique() 
    )

    epoch_list = sorted(epoch_list)
    
    # Initialize indicator_matrix with appropriate size
    label_matrix = np.zeros((len(epoch_list) - 1, 4), dtype=object)

    for i in range(len(epoch_list) - 1):
        # Extract epoch1 and epoch2
        epoch1 = epoch_list[i]
        epoch2 = epoch_list[i+1]

        label_matrix[i,0] = epoch1
        label_matrix[i,1] = epoch2

        # Filter data for the two epochs
        epoch_data1 = data_gps_c1c[data_gps_c1c["time_of_reception_in_receiver_time"] == epoch1]
        epoch_data2 = data_gps_c1c[data_gps_c1c["time_of_reception_in_receiver_time"] == epoch2]

        # Find common satellites between the two epochs
        sat_epoch1 = set(epoch_data1["prn"])
        sat_epoch2 = set(epoch_data2["prn"])
        common_sats = sorted(sat_epoch1 & sat_epoch2)  # Intersection of satellites

        # Filter data to only include common satellites
        epoch_data1 = epoch_data1[epoch_data1["prn"].isin(common_sats)]
        epoch_data2 = epoch_data2[epoch_data2["prn"].isin(common_sats)]

        # Compute indicator for epoch_data1
        sum_LLI1 = epoch_data1["LLI"].sum()
        if sum_LLI1 == 0:
            label_matrix[i, 2] = "no"
        elif sum_LLI1 == 1:
            label_matrix[i, 2] = "single"
        else:
            label_matrix[i, 2] = "multiple"

        # Compute indicator for epoch_data2
        sum_LLI2 = epoch_data2["LLI"].sum()
        if sum_LLI2 == 0:
            label_matrix[i, 3] = "no"  # If LLI == 0
        elif sum_LLI2 == 1:
            label_matrix[i,3] = "single"
            if label_matrix[i, 2] == "single":  # If the label of epoch 1 is single
                prn_fault_epoch2 = epoch_data2[epoch_data2["LLI"] == 1]["prn"].iloc[0] 
                prn_fault_epoch1 = epoch_data1[epoch_data1["LLI"] == 1]["prn"].iloc[0]

                if prn_fault_epoch2 != prn_fault_epoch1:  # If it's the same faulty sat, consider this case
                    label_matrix[i, 2] = f"no fault if exclude SAT {prn_fault_epoch1} in previous epoch"
        else:
            label_matrix[i, 3] = "multiple"

    return label_matrix
