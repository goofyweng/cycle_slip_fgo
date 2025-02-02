import numpy as np
from filter_epoch_fnc import select_evenly_dist_true



if __name__ == "__main__":
    # Example boolean numpy array with 15 'True' and 5 'False'
    bool_array = np.array([True] * 15 + [False] * 5)

    # Shuffle the array for randomness (optional)
    np.random.shuffle(bool_array)

    # Select evenly distributed true in the bool_array
    new_bool_array = select_evenly_dist_true(bool_array=bool_array, num_true=7)

    # Print the results
    print("Original Array:")
    print(bool_array)
    print("\nNew Array with 7 'True' values:")
    print(new_bool_array)