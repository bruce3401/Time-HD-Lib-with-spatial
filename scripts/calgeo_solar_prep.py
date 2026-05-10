"""Prep NASA POWER daily surface solar irradiance for California.

Source: NASA POWER (Prediction Of Worldwide Energy Resources) Daily API.
  https://power.larc.nasa.gov/api/temporal/daily/point
  No API key needed. Stable endpoint.

We sample N stratified points across CA (5 latitude stripes) and pull
ALLSKY_SFC_SW_DWN (all-sky surface shortwave downward irradiance, kWh/m²/day)
for each point at daily resolution.

Output (mirrors AirQuality format):
  dataset/CalGeo/Solar/ghi_flow.csv     (date, point_1, point_2, ...)
  dataset/CalGeo/Solar/ghi_meta.csv     (site_id, lat, lng, county_fips)
  dataset/CalGeo/Solar/ghi_adj.npy      (N, N) KNN-8 adjacency

NASA POWER doesn't have a separate "solar farm" entity — every land grid
cell is queryable. We sample 150 stratified points to mirror NSRDB-style
spatial diversity (coast / valley / desert / mountain / inland).

Usage:
  python scripts/calgeo_solar_prep.py --years 2018 2019 2020 --n_points 150
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

# California land bbox (rough)
CA_BBOX = dict(south=32.5, north=42.0, west=-124.5, east=-114.0)

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def stratified_grid(n_points: int, bbox=CA_BBOX, seed: int = 2024) -> pd.DataFrame:
    """5 lat stripes × uniform lng within stripe; total ≈ n_points."""
    rng = np.random.default_rng(seed)
    n_stripes = 5
    per_stripe = n_points // n_stripes
    rows = []
    for s in range(n_stripes):
        lat_lo = bbox["south"] + (bbox["north"] - bbox["south"]) * s / n_stripes
        lat_hi = bbox["south"] + (bbox["north"] - bbox["south"]) * (s + 1) / n_stripes
        for _ in range(per_stripe):
            lat = rng.uniform(lat_lo, lat_hi)
            lng = rng.uniform(bbox["west"], bbox["east"])
            rows.append((lat, lng))
    df = pd.DataFrame(rows, columns=["lat", "lng"])
    df["site_id"] = [f"NPOWER_{i:04d}" for i in range(len(df))]
    return df[["site_id", "lat", "lng"]]


def fetch_point(lat: float, lng: float, start: str, end: str) -> pd.Series:
    """Fetch daily ALLSKY_SFC_SW_DWN for one point.
    start/end are 'YYYYMMDD'. Returns Series indexed by date."""
    params = {
        "parameters": "ALLSKY_SFC_SW_DWN",
        "community": "RE",
        "longitude": f"{lng:.4f}",
        "latitude": f"{lat:.4f}",
        "start": start,
        "end": end,
        "format": "JSON",
    }
    r = requests.get(NASA_POWER_URL, params=params, timeout=120)
    r.raise_for_status()
    data = r.json()
    series_dict = data["properties"]["parameter"]["ALLSKY_SFC_SW_DWN"]
    s = pd.Series(series_dict, dtype=float)
    s.index = pd.to_datetime(s.index, format="%Y%m%d")
    s.name = "ghi"
    # NASA POWER fill value is -999 for missing; replace with NaN
    s[s <= -990] = np.nan
    return s


def fetch_all(grid: pd.DataFrame, years: list, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    start = f"{min(years)}0101"
    end = f"{max(years)}1231"
    series = {}
    for i, row in grid.iterrows():
        pid = row["site_id"]
        cache = cache_dir / f"{pid}.csv"
        if cache.exists():
            s = pd.read_csv(cache, parse_dates=["date"], index_col="date")["ghi"]
        else:
            print(f"  [{i+1}/{len(grid)}] fetching {pid} ({row.lat:.3f},{row.lng:.3f})")
            s = fetch_point(row.lat, row.lng, start, end)
            s.to_frame("ghi").rename_axis("date").to_csv(cache)
            time.sleep(0.2)  # gentle rate limit
        series[pid] = s
    return pd.DataFrame(series).sort_index()


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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", nargs="+", type=int, default=[2018, 2019, 2020])
    ap.add_argument("--n_points", type=int, default=150)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--cache_dir", default="dataset/CalGeo/Solar/_cache")
    ap.add_argument("--out_dir", default="dataset/CalGeo/Solar")
    args = ap.parse_args()

    grid = stratified_grid(args.n_points)
    print(f"[solar/NASA-POWER] sampling {len(grid)} points across CA, years {args.years}")

    cache_dir = Path(args.cache_dir).resolve()
    out_dir = Path(args.out_dir).resolve()
    wide = fetch_all(grid, args.years, cache_dir)
    print(f"[solar] wide shape (dates × points): {wide.shape}")

    # Drop points with too many NaN; fill remaining gaps with column mean
    coverage = (1.0 - wide.isna().mean(axis=0))
    keep = coverage[coverage >= 0.95].index
    wide = wide[keep]
    print(f"[solar] after coverage filter (>=95%): {wide.shape}")
    wide = wide.fillna(wide.mean(axis=0))

    meta = grid[grid["site_id"].isin(wide.columns)].copy().reset_index(drop=True)
    meta["county_fips"] = ""
    wide = wide[meta["site_id"].tolist()]
    A = knn_adj(meta, k=args.knn)
    print(f"[solar] meta rows: {len(meta)}, adj density: {A.sum() / A.size:.4f}")

    out_dir.mkdir(parents=True, exist_ok=True)
    wide.rename_axis("date").to_csv(out_dir / "ghi_flow.csv")
    meta.to_csv(out_dir / "ghi_meta.csv", index=False)
    np.save(out_dir / "ghi_adj.npy", A)
    print(f"[done] solar prep wrote {out_dir}/ghi_*")


if __name__ == "__main__":
    main()
