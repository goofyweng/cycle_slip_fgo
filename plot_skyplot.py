import numpy as np
import matplotlib.pyplot as plt

def plot_skyplot(satellite_prns, satellite_positions, user_position, epoch=None):
    """
    Plots a skyplot for visible satellites.

    Args:
        satellite_prns (np.array of int): Array of satellite PRN numbers (shape: N,).
        satellite_positions (np.array): Satellite positions in ECEF (shape: Nx3).
        user_position (np.array): User position in ECEF (shape: 3,).
        epoch: The epoch of the current epoch, default None.

    Returns:
        None: Displays the skyplot.
    """
    def ecef_to_enu(sat_ecef, user_ecef):
        # Convert ECEF to ENU coordinates
        rel_pos = sat_ecef - user_ecef

        # Compute user latitude and longitude
        user_lat = np.arctan2(user_ecef[2], np.sqrt(user_ecef[0]**2 + user_ecef[1]**2))
        user_lon = np.arctan2(user_ecef[1], user_ecef[0])

        # Rotation matrices
        R1 = np.array([[-np.sin(user_lon), np.cos(user_lon), 0],
                       [-np.sin(user_lat) * np.cos(user_lon), -np.sin(user_lat) * np.sin(user_lon), np.cos(user_lat)],
                       [np.cos(user_lat) * np.cos(user_lon), np.cos(user_lat) * np.sin(user_lon), np.sin(user_lat)]])

        # Transform to ENU
        enu = R1 @ rel_pos
        return enu

    def enu_to_az_el(enu):
        # Compute azimuth and elevation
        e, n, u = enu
        azimuth = np.arctan2(e, n) % (2 * np.pi)  # Azimuth in radians
        elevation = np.arcsin(u / np.linalg.norm(enu))  # Elevation in radians
        return azimuth, elevation

    azimuths = []
    elevations = []

    # Compute azimuth and elevation for each satellite
    for sat_ecef in satellite_positions:
        enu = ecef_to_enu(sat_ecef, user_position)
        az, el = enu_to_az_el(enu)
        azimuths.append(az)
        elevations.append(el)

    # Convert elevations to radial distance for skyplot
    radii = 90 - np.degrees(elevations)

    # Plot the skyplot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')  # North at the top
    ax.set_theta_direction(-1)  # Azimuth increases clockwise
    ax.set_ylim(0, 90)
    ax.set_yticks(range(0, 91, 30))
    ax.set_yticklabels(map(str, [90, 60, 30, 0]))  # Elevation labels
    if epoch != None:
        ax.set_title(f"Skyplot of Visible Satellites at epoch {epoch}", va='bottom')
    else:
        ax.set_title("Skyplot of Visible Satellites", va='bottom')

    # Plot satellites with PRN labels
    for az, r, prn in zip(azimuths, radii, satellite_prns):
        ax.scatter(az, r, s=100, c='cyan')  # Plot satellite point
        ax.text(az, r, str(prn), fontsize=10, ha='center', va='center', color='black')  # Add PRN label

    plt.show()

if __name__ == "__main__":
    # Example Usage
    satellite_prns = np.array([1, 2, 3, 4])  # PRNs as integers
    satellite_positions = np.array([
        [15600e3, 7540e3, 20140e3],
        [18760e3, 2750e3, 17610e3],
        [17610e3, 14630e3, 13480e3],
        [19170e3, 610e3, 17380e3]
    ])
    user_position = np.array([1113e3, 6001e3, 3221e3])

    plot_skyplot(satellite_prns, satellite_positions, user_position)
