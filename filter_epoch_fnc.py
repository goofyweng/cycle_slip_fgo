import pandas as pd
import numpy as np

def send_indicator(data_gps_c1c):

    epoch_list = data_gps_c1c["time_of_reception_in_receiver_time"].unique()
    epoch_list = sorted(epoch_list)
    
    # Initialize indicator_matrix with appropriate size
    indicator_matrix = np.zeros((len(epoch_list) - 1, 4), dtype=object)

    for i in range(len(epoch_list) - 1):
        # Extract epoch1 and epoch2
        epoch1 = epoch_list[i]
        epoch2 = epoch_list[i+1]

        indicator_matrix[i,0] = epoch1
        indicator_matrix[i,1] = epoch2

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
            indicator_matrix[i, 2] = "no"
        elif sum_LLI1 == 1:
            indicator_matrix[i, 2] = "single"
        else:
            indicator_matrix[i, 2] = "multiple"

        # Compute indicator for epoch_data2
        sum_LLI2 = epoch_data2["LLI"].sum()
        if sum_LLI2 == 0:
            indicator_matrix[i, 3] = "no"  # 如果 LLI == 0
        elif sum_LLI2 == 1:
            prn_fault_epoch2 = epoch_data2[epoch_data2["LLI"] == 1]["prn"].iloc[0] # 找出epoch2中LLI為1的衛星PRN
 
            if indicator_matrix[i, 2] == "single":  # 如果epoch1的指標是single
                prn_fault_epoch1 = epoch_data1[epoch_data1["LLI"] == 1]["prn"].iloc[0]  # 找出epoch1中LLI為1的衛星PRN
                
                if prn_fault_epoch2 == prn_fault_epoch1:  # 如果是同一顆衛星
                    indicator_matrix[i, 3] = "single"  # 認為這顆衛星有 cycle slip
                else:
                    indicator_matrix[i, 2] = f"no fault if exclude SAT {prn_fault_epoch1} in previous epoch"
                    indicator_matrix[i, 3] = "single"  # 如果不同衛星，則認為前一刻的衛星沒有錯誤
        else:
            indicator_matrix[i, 3] = "multiple"  # 如果有多顆衛星錯誤

    return indicator_matrix
