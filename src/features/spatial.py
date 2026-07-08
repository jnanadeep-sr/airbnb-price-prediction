"""Location-based feature engineering (distance to the beach, tourist hotspots, etc.)."""
import numpy as np

from src.config import IPANEMA_LAT, IPANEMA_LON, RIO_CENTER_LAT, RIO_CENTER_LON, RIO_HOTSPOTS


def haversine_distance(lat1, lon1, lat2, lon2):
    """Great-circle distance in km between two lat/lon points."""
    R = 6371
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    dphi = np.radians(lat2 - lat1)
    dlambda = np.radians(lon2 - lon1)
    a = np.sin(dphi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(dlambda / 2) ** 2
    return 2 * R * np.arctan2(np.sqrt(a), np.sqrt(1 - a))


def add_beach_distance(df, lat_col="latitude", lon_col="longitude"):
    """Add a 'dist_to_beach' column measuring distance to Ipanema Beach."""
    df = df.copy()
    df["dist_to_beach"] = haversine_distance(df[lat_col], df[lon_col], IPANEMA_LAT, IPANEMA_LON)
    return df


def add_city_center_distance(df, lat_col="latitude", lon_col="longitude"):
    """Add a 'dist_to_city_center' column measuring distance to Centro, Rio de Janeiro."""
    df = df.copy()
    df["dist_to_city_center"] = haversine_distance(df[lat_col], df[lon_col], RIO_CENTER_LAT, RIO_CENTER_LON)
    return df


def add_hotspot_distances(df, lat_col="latitude", lon_col="longitude", hotspots=RIO_HOTSPOTS):
    """Add a distance column per Rio landmark plus an averaged 'centrality_score'."""
    df = df.copy()
    for name, (hot_lat, hot_lon) in hotspots.items():
        df[name] = haversine_distance(df[lat_col], df[lon_col], hot_lat, hot_lon)
    df["centrality_score"] = df[list(hotspots.keys())].mean(axis=1)
    return df
