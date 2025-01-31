import numpy as np
import matplotlib.pyplot as plt

def plot_skyplot(satellite_positions, user_position, satellite_prns=np.array([]), epoch=None):
    """
    Plots a skyplot for visible satellites.

    Args:
        satellite_positions (np.array): Satellite positions in ECEF coordinates (shape: Nx3).
        user_position (np.array): User position in ECEF coordinates (shape: 3,).
        satellite_prns (np.array of int, optional): Array of satellite PRN numbers (shape: N,). Defaults to an empty array.
        epoch (optional): The current epoch, default is None.

    Returns:
        None: Displays the skyplot.
    """

    def ecef_to_enu(sat_ecef, user_ecef):
        """
        Converts ECEF coordinates to ENU (East-North-Up) coordinates.

        Args:
            sat_ecef (np.array): Satellite position in ECEF (shape: 3,).
            user_ecef (np.array): User position in ECEF (shape: 3,).

        Returns:
            np.array: ENU coordinates of the satellite.
        """
        rel_pos = sat_ecef - user_ecef
        user_lat = np.arctan2(user_ecef[2], np.sqrt(user_ecef[0]**2 + user_ecef[1]**2))
        user_lon = np.arctan2(user_ecef[1], user_ecef[0])

        # Rotation matrix to convert ECEF to ENU
        R1 = np.array([
            [-np.sin(user_lon), np.cos(user_lon), 0],
            [-np.sin(user_lat) * np.cos(user_lon), -np.sin(user_lat) * np.sin(user_lon), np.cos(user_lat)],
            [np.cos(user_lat) * np.cos(user_lon), np.cos(user_lat) * np.sin(user_lon), np.sin(user_lat)]
        ])
        return R1 @ rel_pos

    def enu_to_az_el(enu):
        """
        Converts ENU coordinates to azimuth and elevation.

        Args:
            enu (np.array): ENU coordinates (shape: 3,).

        Returns:
            tuple: (azimuth in radians, elevation in radians).
        """
        e, n, u = enu
        azimuth = np.arctan2(e, n) % (2 * np.pi)  # Normalize azimuth to [0, 2π]
        elevation = np.arcsin(u / np.linalg.norm(enu))  # Compute elevation angle
        return azimuth, elevation

    azimuths = []
    elevations = []

    # Compute azimuth and elevation for each satellite
    for sat_ecef in satellite_positions:
        enu = ecef_to_enu(sat_ecef, user_position)
        az, el = enu_to_az_el(enu)
        azimuths.append(az)
        elevations.append(el)

    # Convert elevation to polar plot radius (90° at center, 0° at outer edge)
    radii = 90 - np.degrees(elevations)

    # Create a polar plot for the skyplot
    fig, ax = plt.subplots(subplot_kw={'projection': 'polar'})
    ax.set_theta_zero_location('N')  # Set north as the reference direction
    ax.set_theta_direction(-1)  # Set azimuth to increase clockwise
    ax.set_ylim(0, 90)  # Set elevation limits (0° at outermost, 90° at center)
    ax.set_yticks(range(0, 91, 30))
    ax.set_yticklabels(map(str, [90, 60, 30, 0]))  # Label elevation circles

    # Set plot title with epoch information if provided
    title_text = f"Skyplot of Visible Satellites at epoch {epoch}" if epoch is not None else "Skyplot of Visible Satellites"
    ax.set_title(title_text, va='bottom')

    # Plot satellite positions (even if PRN is not provided)
    ax.scatter(azimuths, radii, s=100, c='cyan')

    # If PRN numbers are provided, annotate each satellite
    if satellite_prns.size > 0:
        for az, r, prn in zip(azimuths, radii, satellite_prns):
            ax.text(az, r, str(prn), fontsize=10, ha='center', va='center', color='black')

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
