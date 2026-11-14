
import numpy as np
import pandas as pd

KM_PER_DEG = 111.195
FARE_HIKE_DATE = pd.Timestamp("2012-09-04")
START_DATE = pd.Timestamp("2009-01-01")

# landmark -> (latitude, longitude)
LANDMARKS = {
    "jfk":      (40.6413, -73.7781),
    "lga":      (40.7769, -73.8740),
    "ewr":      (40.6895, -74.1745),
    "times_sq": (40.7580, -73.9855),
    "wall_st":  (40.7060, -74.0088),
}
AIRPORT_RADIUS_KM = {"jfk": 3.0, "lga": 2.0, "ewr": 3.0}

FEATURES = (
    ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude", "passenger_count",
     "dlat_km", "dlon_km", "dist_km", "manhattan_km", "bearing_sin", "bearing_cos", "dist_per_passenger",
     "hour", "dayofweek", "month", "year", "years_since_2009",
     "is_weekend", "is_night", "is_rush_hour", "post_fare_hike",
     "hour_sin", "hour_cos", "dow_sin", "dow_cos"]
    + ["pu_dist_" + n for n in LANDMARKS]
    + ["do_dist_" + n for n in LANDMARKS]
    + ["pickup_airport", "dropoff_airport", "airport_trip"]
)

# features obtained directly from the original columns (used as the "raw" baseline set)
RAW_FEATURES = ["pickup_longitude", "pickup_latitude", "dropoff_longitude", "dropoff_latitude",
                "passenger_count", "hour", "dayofweek", "month", "year"]


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0088
    p1, p2 = np.radians(lat1), np.radians(lat2)
    a = (np.sin((p2 - p1) / 2) ** 2
         + np.cos(p1) * np.cos(p2) * np.sin(np.radians(lon2 - lon1) / 2) ** 2)
    return 2 * R * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def make_features(raw):
    # raw needs: pickup_datetime, pickup/dropoff longitude and latitude, passenger_count
    ts = pd.to_datetime(raw["pickup_datetime"])
    if ts.dt.tz is not None:
        ts = ts.dt.tz_localize(None)          # keep the clock time (it is NYC local time)

    plat = raw["pickup_latitude"].to_numpy("float64")
    plon = raw["pickup_longitude"].to_numpy("float64")
    dlat = raw["dropoff_latitude"].to_numpy("float64")
    dlon = raw["dropoff_longitude"].to_numpy("float64")
    pax = np.maximum(raw["passenger_count"].to_numpy("float64"), 1.0)

    f = {}
    f["pickup_longitude"], f["pickup_latitude"] = plon, plat
    f["dropoff_longitude"], f["dropoff_latitude"] = dlon, dlat
    f["passenger_count"] = pax

    # displacement, distance, direction
    f["dlat_km"] = (dlat - plat) * KM_PER_DEG
    f["dlon_km"] = (dlon - plon) * KM_PER_DEG * np.cos(np.radians((plat + dlat) / 2))
    f["dist_km"] = haversine_km(plat, plon, dlat, dlon)
    f["manhattan_km"] = np.abs(f["dlat_km"]) + np.abs(f["dlon_km"])
    ang = np.arctan2(f["dlon_km"], f["dlat_km"])          # compass bearing, 0 = north
    moved = f["dist_km"] > 1e-3
    f["bearing_sin"] = np.where(moved, np.sin(ang), 0.0)
    f["bearing_cos"] = np.where(moved, np.cos(ang), 0.0)
    f["dist_per_passenger"] = f["dist_km"] / pax

    # time
    hour = ts.dt.hour.to_numpy()
    dow = ts.dt.dayofweek.to_numpy()
    f["hour"], f["dayofweek"] = hour, dow
    f["month"], f["year"] = ts.dt.month.to_numpy(), ts.dt.year.to_numpy()
    f["years_since_2009"] = (ts - START_DATE).dt.total_seconds().to_numpy() / (365.25 * 86400)
    f["is_weekend"] = dow >= 5
    f["is_night"] = (hour >= 20) | (hour < 6)
    f["is_rush_hour"] = (dow < 5) & (hour >= 16) & (hour < 20)
    f["post_fare_hike"] = (ts >= FARE_HIKE_DATE).to_numpy()
    f["hour_sin"], f["hour_cos"] = np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24)
    f["dow_sin"], f["dow_cos"] = np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7)

    # location
    for name, (la, lo) in LANDMARKS.items():
        f["pu_dist_" + name] = haversine_km(plat, plon, la, lo)
        f["do_dist_" + name] = haversine_km(dlat, dlon, la, lo)
    f["pickup_airport"] = np.zeros(len(raw), dtype=bool)
    f["dropoff_airport"] = np.zeros(len(raw), dtype=bool)
    for name, r in AIRPORT_RADIUS_KM.items():
        f["pickup_airport"] |= f["pu_dist_" + name] < r
        f["dropoff_airport"] |= f["do_dist_" + name] < r
    f["airport_trip"] = f["pickup_airport"] | f["dropoff_airport"]

    X = pd.DataFrame(f, index=raw.index)
    for c in X.columns:
        k = X[c].dtype.kind
        if k == "b":
            X[c] = X[c].astype("int8")
        elif k == "f":
            X[c] = X[c].astype("float32")
        elif k == "i":
            X[c] = X[c].astype("int16")
    return X[FEATURES]
