"""Prep NOAA GHCN-Daily weather stations for California into Time-HD format.

Source: NOAA GHCN-D (Global Historical Climatology Network - Daily).
  Public S3 bucket: https://noaa-ghcn-pds.s3.amazonaws.com/
  No API key needed.

We pull California stations (state code "CA" in ghcnd-stations.txt) with
sufficient coverage in our date range, and extract daily TMAX/TMIN/PRCP.

Output (mirrors AirQuality format):
  dataset/CalGeo/Weather/tmax_flow.csv     (date, station_1, ...)
  dataset/CalGeo/Weather/tmax_meta.csv     (station_id, lat, lng, county_fips)
  dataset/CalGeo/Weather/tmax_adj.npy      (N, N) KNN(k=8)

  Same for tmin, prcp.

Usage:
  python scripts/calgeo_weather_prep.py --years 2018 2019 2020 --vars TMAX TMIN PRCP
"""
from __future__ import annotations
import argparse
import io
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import requests

GHCN_S3 = "https://noaa-ghcn-pds.s3.amazonaws.com"
STATIONS_URL = f"{GHCN_S3}/ghcnd-stations.txt"
DATA_BY_YEAR = f"{GHCN_S3}/csv.gz"  # /csv.gz/<YYYY>.csv.gz

# Variable scales (GHCN-D stores fixed-point):
# TMAX, TMIN: 0.1°C; PRCP: 0.1 mm; SNOW: 1 mm; SNWD: 1 mm
VAR_SCALE = {
    "TMAX": 0.1,
    "TMIN": 0.1,
    "PRCP": 0.1,
    "SNOW": 1.0,
    "SNWD": 1.0,
}


def load_stations(cache_dir: Path) -> pd.DataFrame:
    """Parse ghcnd-stations.txt fixed-width to a DataFrame, filter to CA."""
    cache = cache_dir / "ghcnd-stations.txt"
    if not cache.exists():
        cache.parent.mkdir(parents=True, exist_ok=True)
        print(f"  fetching {STATIONS_URL}")
        r = requests.get(STATIONS_URL, timeout=120); r.raise_for_status()
        cache.write_bytes(r.content)
    # Fixed-width spec from the GHCN-D readme
    cols = [
        ("station_id", 0, 11),
        ("lat", 12, 20),
        ("lng", 21, 30),
        ("elev", 31, 37),
        ("state", 38, 40),
        ("name", 41, 71),
    ]
    rows = []
    with open(cache) as fh:
        for line in fh:
            row = {c[0]: line[c[1]:c[2]].strip() for c in cols}
            rows.append(row)
    df = pd.DataFrame(rows)
    df["lat"] = df["lat"].astype(float)
    df["lng"] = df["lng"].astype(float)
    return df


def filter_us_ca(stations: pd.DataFrame) -> pd.DataFrame:
    """US stations have ID prefix 'US'; CA stations have state == 'CA'."""
    us = stations[stations["station_id"].str.startswith(("US",))]
    ca = us[us["state"] == "CA"].copy().reset_index(drop=True)
    return ca


def fetch_year(year: int, cache_dir: Path) -> pd.DataFrame:
    """Download one year's full GHCN-D records (CSV.gz) and return as DataFrame.

    Schema: ID, DATE(YYYYMMDD), ELEMENT, VALUE, M-FLAG, Q-FLAG, S-FLAG, OBS-TIME.
    Only keep rows with empty Q-FLAG (good quality)."""
    cache = cache_dir / f"{year}.csv.gz"
    if not cache.exists():
        url = f"{DATA_BY_YEAR}/{year}.csv.gz"
        print(f"  fetching {url}")
        r = requests.get(url, timeout=300); r.raise_for_status()
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(r.content)
    df = pd.read_csv(cache, header=None,
                     names=["station_id","date","element","value","mflag","qflag","sflag","obstime"],
                     dtype={"station_id":"string","element":"string"})
    # quality filter: keep rows with no qflag (NaN or empty string)
    df = df[df["qflag"].isna() | (df["qflag"].astype(str).str.strip() == "")]
    df["date"] = pd.to_datetime(df["date"].astype(str), format="%Y%m%d")
    return df


def reshape_one_var(df: pd.DataFrame, ca_ids: set, var: str) -> pd.DataFrame:
    """For one ELEMENT, pivot to (date × station_id), apply scale."""
    sub = df[(df["element"] == var) & (df["station_id"].isin(ca_ids))].copy()
    sub["value"] = sub["value"].astype(float) * VAR_SCALE[var]
    pivot = sub.pivot_table(index="date", columns="station_id",
                             values="value", aggfunc="mean")
    pivot.index.name = "date"
    return pivot


def coverage_filter(wide: pd.DataFrame, min_coverage: float) -> pd.DataFrame:
    coverage = (1.0 - wide.isna().mean(axis=0))
    keep = coverage[coverage >= min_coverage].index
    return wide[keep]


def fill_gaps(wide: pd.DataFrame) -> pd.DataFrame:
    return wide.fillna(wide.mean(axis=0))


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    R = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1; dlng = lng2 - lng1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def knn_adj(meta: pd.DataFrame, k: int = 8) -> np.ndarray:
    N = len(meta)
    lat = meta["lat"].to_numpy(); lng = meta["lng"].to_numpy()
    A = np.zeros((N, N), dtype=np.float32)
    LAT = np.broadcast_to(lat, (N, N)); LNG = np.broadcast_to(lng, (N, N))
    D = haversine_km(LAT, LNG, LAT.T, LNG.T)
    np.fill_diagonal(D, np.inf)
    nearest = np.argsort(D, axis=1)[:, :k]
    for i in range(N):
        A[i, nearest[i]] = 1.0
    return ((A + A.T) > 0).astype(np.float32)


def prep_one_var(var: str, frames: list, ca_ids: set, ca_meta: pd.DataFrame,
                 out_dir: Path, k: int, min_coverage: float):
    print(f"[{var}] reshaping {len(frames)} year-frames")
    wides = [reshape_one_var(f, ca_ids, var) for f in frames]
    wide = pd.concat(wides).sort_index()
    print(f"  raw shape: {wide.shape}")
    wide = coverage_filter(wide, min_coverage)
    print(f"  after coverage filter (≥{int(min_coverage*100)}%): {wide.shape}")
    wide = fill_gaps(wide)
    meta = ca_meta[ca_meta["station_id"].isin(wide.columns)].copy()
    meta = meta.rename(columns={"station_id": "site_id"})  # standardize with calgeo.py loader
    meta = meta[["site_id", "lat", "lng"]]
    meta["county_fips"] = ""
    wide = wide[meta["site_id"].tolist()]
    A = knn_adj(meta, k=k)
    out_dir.mkdir(parents=True, exist_ok=True)
    wide.to_csv(out_dir / f"{var.lower()}_flow.csv")
    meta.to_csv(out_dir / f"{var.lower()}_meta.csv", index=False)
    np.save(out_dir / f"{var.lower()}_adj.npy", A)
    print(f"  wrote {var.lower()}_flow.csv / {var.lower()}_meta.csv / {var.lower()}_adj.npy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    ap.add_argument("--vars", nargs="+", default=["TMAX", "TMIN", "PRCP"],
                    choices=list(VAR_SCALE.keys()))
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--min_coverage", type=float, default=0.7)
    ap.add_argument("--cache_dir", default="dataset/CalGeo/Weather/_cache")
    ap.add_argument("--out_dir", default="dataset/CalGeo/Weather")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    print("[weather] loading station catalogue")
    stations = load_stations(cache_dir)
    ca = filter_us_ca(stations)
    ca_ids = set(ca["station_id"])
    print(f"  CA stations in catalogue: {len(ca_ids)}")

    print("[weather] downloading + filtering yearly records")
    frames = [fetch_year(y, cache_dir) for y in args.years]

    for var in args.vars:
        prep_one_var(var, frames, ca_ids, ca, out_dir, args.knn, args.min_coverage)

    print("[done] Weather prep complete.")


if __name__ == "__main__":
    main()
