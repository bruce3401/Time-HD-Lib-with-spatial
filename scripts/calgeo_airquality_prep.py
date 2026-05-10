"""Prep EPA AQS daily PM2.5 + Ozone for California into Time-HD format.

Source: EPA Air Quality System (AQS) public CSV downloads.
  https://aqs.epa.gov/aqsweb/airdata/download_files.html

Pulls yearly daily-summary files:
  daily_88101_<year>.zip   PM2.5 (FRM/FEM Mass), parameter 88101
  daily_44201_<year>.zip   Ozone, parameter 44201

Filters to California (state code 06), pivots to (date × site_id) wide,
computes station coords + KNN(k=8) adjacency by Haversine distance.

Output:
  dataset/CalGeo/AirQuality/pm25_flow.csv     (date, site_1, site_2, ...)
  dataset/CalGeo/AirQuality/pm25_meta.csv     (site_id, lat, lng, county_fips)
  dataset/CalGeo/AirQuality/pm25_adj.npy      (N, N) KNN-8 binary (symmetric)
  (same triplet for ozone)

Usage:
  python scripts/calgeo_airquality_prep.py --years 2018 2019 2020
"""
from __future__ import annotations
import argparse
import io
import os
import sys
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd
import requests

CA_STATE_CODE = "06"

# Parameter codes per EPA AQS
PARAM = {
    "pm25":  ("88101", "PM2.5 - Mass (FRM/FEM)"),
    "ozone": ("44201", "Ozone (8-hr daily max)"),
}

EPA_BASE = "https://aqs.epa.gov/aqsweb/airdata"


def download_year(param_code: str, year: int, cache_dir: Path) -> pd.DataFrame:
    """Download daily summary CSV for one parameter+year. Cache to disk."""
    cache_csv = cache_dir / f"daily_{param_code}_{year}.csv"
    if cache_csv.exists():
        return pd.read_csv(cache_csv, dtype={"State Code": str, "County Code": str, "Site Num": str})

    url = f"{EPA_BASE}/daily_{param_code}_{year}.zip"
    print(f"  fetching {url}")
    r = requests.get(url, timeout=120)
    r.raise_for_status()
    with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
        # the zip contains a single CSV with the matching name
        member = [m for m in zf.namelist() if m.endswith(".csv")][0]
        with zf.open(member) as fh:
            df = pd.read_csv(fh, dtype={"State Code": str, "County Code": str, "Site Num": str})
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(cache_csv, index=False)
    return df


def filter_ca(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df["State Code"] == CA_STATE_CODE].copy()
    # AQS site identifier: state.county.site
    df["site_id"] = (df["State Code"] + df["County Code"].str.zfill(3)
                     + df["Site Num"].str.zfill(4))
    df["county_fips"] = df["State Code"] + df["County Code"].str.zfill(3)
    df["Date Local"] = pd.to_datetime(df["Date Local"])
    return df


def reshape_to_wide(df: pd.DataFrame, value_col: str = "Arithmetic Mean") -> pd.DataFrame:
    """One row per date, columns are site_ids. Average duplicate (site, date) pairs."""
    pivot = df.pivot_table(index="Date Local", columns="site_id",
                           values=value_col, aggfunc="mean")
    pivot.index.name = "date"
    return pivot


def build_meta(df: pd.DataFrame) -> pd.DataFrame:
    """One row per site_id with lat/lng/county_fips (taking median over years)."""
    meta = (df.groupby("site_id")
              .agg(lat=("Latitude", "median"),
                   lng=("Longitude", "median"),
                   county_fips=("county_fips", "first"))
              .reset_index())
    return meta


def haversine_km(lat1, lng1, lat2, lng2) -> np.ndarray:
    """Vectorized Haversine, all inputs in degrees, returns km."""
    R = 6371.0
    lat1, lng1, lat2, lng2 = map(np.radians, (lat1, lng1, lat2, lng2))
    dlat = lat2 - lat1; dlng = lng2 - lng1
    a = np.sin(dlat/2.0)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng/2.0)**2
    return 2 * R * np.arcsin(np.sqrt(a))


def knn_adj(meta: pd.DataFrame, k: int = 8) -> np.ndarray:
    """Symmetric KNN-k binary adjacency by Haversine."""
    N = len(meta)
    lat = meta["lat"].to_numpy(); lng = meta["lng"].to_numpy()
    A = np.zeros((N, N), dtype=np.float32)
    # pairwise distance
    LAT = np.broadcast_to(lat, (N, N))
    LNG = np.broadcast_to(lng, (N, N))
    D = haversine_km(LAT, LNG, LAT.T, LNG.T)
    np.fill_diagonal(D, np.inf)
    nearest = np.argsort(D, axis=1)[:, :k]
    for i in range(N):
        A[i, nearest[i]] = 1.0
    return ((A + A.T) > 0).astype(np.float32)


def coverage_filter(wide: pd.DataFrame, min_coverage: float = 0.7) -> pd.DataFrame:
    """Keep only sites that report on ≥ min_coverage fraction of the date range."""
    n_dates = len(wide)
    coverage = (1.0 - wide.isna().mean(axis=0))
    keep = coverage[coverage >= min_coverage].index
    return wide[keep]


def fill_gaps(wide: pd.DataFrame) -> pd.DataFrame:
    """Fill NaN with column-mean (so the time series has no holes)."""
    means = wide.mean(axis=0)
    return wide.fillna(means)


def prep_one_param(param_key: str, years: list, cache_dir: Path,
                   out_dir: Path, k: int, min_coverage: float):
    code, name = PARAM[param_key]
    print(f"[{param_key}] downloading + assembling {name} ({code}) for years {years}")
    frames = []
    for y in years:
        df = download_year(code, y, cache_dir / param_key)
        df = filter_ca(df)
        frames.append(df)
    df_all = pd.concat(frames, ignore_index=True)
    print(f"  CA daily rows total: {len(df_all):,}")

    wide = reshape_to_wide(df_all)
    print(f"  wide shape (dates × sites): {wide.shape}")

    wide = coverage_filter(wide, min_coverage)
    print(f"  after coverage filter (≥{int(min_coverage*100)}%): {wide.shape}")

    wide = fill_gaps(wide)
    meta = build_meta(df_all)
    meta = meta[meta["site_id"].isin(wide.columns)].reset_index(drop=True)
    # Re-order wide columns to match meta site_id order
    wide = wide[meta["site_id"].tolist()]

    A = knn_adj(meta, k=k)
    print(f"  meta rows: {len(meta)}, adj density: {A.sum() / A.size:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    flow_path = out_dir / f"{param_key}_flow.csv"
    meta_path = out_dir / f"{param_key}_meta.csv"
    adj_path  = out_dir / f"{param_key}_adj.npy"
    wide.to_csv(flow_path)
    meta.to_csv(meta_path, index=False)
    np.save(adj_path, A)
    print(f"  wrote {flow_path.name} / {meta_path.name} / {adj_path.name}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    ap.add_argument("--params", nargs="+", default=["pm25", "ozone"],
                    choices=list(PARAM.keys()))
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--min_coverage", type=float, default=0.7,
                    help="drop sites reporting on <X fraction of date range")
    ap.add_argument("--cache_dir", default="dataset/CalGeo/AirQuality/_cache")
    ap.add_argument("--out_dir", default="dataset/CalGeo/AirQuality")
    args = ap.parse_args()

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()

    for p in args.params:
        prep_one_param(p, args.years, cache_dir, out_dir, args.knn, args.min_coverage)

    print("[done] AirQuality prep complete.")


if __name__ == "__main__":
    main()
