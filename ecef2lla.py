import numpy as np

def ecef2lla(x, y, z):
    # WGS84 ellipsoid constants:
    a = 6378137.0
    e = 8.1819190842622e-2
    
    # Derived constants
    b = np.sqrt(a**2 * (1 - e**2))
    ep = np.sqrt((a**2 - b**2) / b**2)
    
    # Calculations
    p = np.sqrt(x**2 + y**2)
    th = np.arctan2(a * z, b * p)
    lon = np.arctan2(y, x)
    lat = np.arctan2((z + ep**2 * b * np.sin(th)**3), 
                     (p - e**2 * a * np.cos(th)**3))
    N = a / np.sqrt(1 - e**2 * np.sin(lat)**2)
    alt = p / np.cos(lat) - N
    
    # Ensure longitude is in the range [0, 2*pi)
    lon = np.mod(lon, 2 * np.pi)
    
    # Correct for numerical instability in altitude near exact poles:
    k = (np.abs(x) < 1) & (np.abs(y) < 1)
    alt[k] = np.abs(z[k]) - b
    
    return lat, lon, alt
