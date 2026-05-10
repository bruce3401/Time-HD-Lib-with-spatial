"""Reshape Mobility_CA's 4 long-format NAICS CSVs into Time-HD-format
(weeks × tracts) matrices, joined to TIGER 2020 CA tract geometry for
centroid lat/lng + queen-contiguity adjacency.

Output per category:
  dataset/Mobility_CA/processed/<category>_flow.csv    (date, GEOID1, GEOID2, ...)
  dataset/Mobility_CA/processed/<category>_meta.csv    (GEOID, lat, lng, county_fips, area_sqkm)
  dataset/Mobility_CA/processed/<category>_adj.npy     (N, N) queen contiguity (binary)
"""
from __future__ import annotations
import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

CATS = {
    "essential_retail": "essential_retail_tract.csv",
    "food_and_drink": "food_and_drink_tract.csv",
    "indoor_discretionary": "indoor_discretionary_tract.csv",
    "outdoor_discretionary": "outdoor_discretionary_tract.csv",
}

CA_STATE_FIPS = "06"


def load_tiger(tiger_shp: str, target: str = "EPSG:4326"):
    import geopandas as gpd
    g = gpd.read_file(tiger_shp).to_crs(target)
    g = g[["GEOID", "COUNTYFP", "INTPTLAT", "INTPTLON", "ALAND", "geometry"]].copy()
    g["GEOID"] = g["GEOID"].astype(str).str.zfill(11)
    g["INTPTLAT"] = g["INTPTLAT"].astype(float)
    g["INTPTLON"] = g["INTPTLON"].astype(float)
    g["area_sqkm"] = g["ALAND"].astype(float) / 1e6
    return g


def queen_adj(g) -> np.ndarray:
    """Pairwise queen contiguity matrix in the order of g['GEOID']."""
    from libpysal.weights import Queen
    w = Queen.from_dataframe(g, use_index=False)
    n = len(g)
    A = np.zeros((n, n), dtype=np.float32)
    for i, neighbors in w.neighbor_offsets.items():
        for j in neighbors:
            A[i, j] = 1.0
    A = (A + A.T) > 0
    return A.astype(np.float32)


def queen_adj_fallback(g) -> np.ndarray:
    """If libpysal is missing, fall back to shapely-based pairwise touches.
    O(N^2) intersect — slow on N>5k but fine for one-off prep."""
    n = len(g)
    A = np.zeros((n, n), dtype=np.float32)
    geoms = list(g.geometry.values)
    # Use a spatial index to limit candidates
    from shapely.strtree import STRtree
    tree = STRtree(geoms)
    for i, gi in enumerate(geoms):
        # query candidates whose bounding boxes intersect
        idx = tree.query(gi)
        for j in idx:
            j = int(j)
            if j == i:
                continue
            if gi.touches(geoms[j]) or gi.intersects(geoms[j]):
                A[i, j] = 1.0
                A[j, i] = 1.0
    return A


def reshape_category(csv_path: str, ca_geoids: set) -> pd.DataFrame:
    """Long → wide pivot: (date × GEOID) sum of total_visits."""
    df = pd.read_csv(csv_path, dtype={"tract_geoid": str},
                     usecols=["DATE_RANGE_START", "tract_geoid", "total_visits"])
    df["tract_geoid"] = df["tract_geoid"].str.zfill(11)
    # Keep only California tracts
    df = df[df["tract_geoid"].isin(ca_geoids)]
    df["DATE_RANGE_START"] = pd.to_datetime(df["DATE_RANGE_START"])
    pivot = (df.groupby(["DATE_RANGE_START", "tract_geoid"])["total_visits"]
               .sum()
               .unstack(fill_value=0))
    pivot.index.name = "date"
    pivot = pivot.sort_index()
    return pivot.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="dataset/Mobility_CA")
    ap.add_argument("--out", default=None)
    ap.add_argument("--categories", nargs="+", default=list(CATS.keys()))
    ap.add_argument("--skip_adj", action="store_true",
                    help="Skip queen-adjacency computation (slow on large N).")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.out or (root / "processed"))
    out.mkdir(parents=True, exist_ok=True)

    tiger_shp = root / "TIGER" / "tl_2020_06_tract.shp"
    if not tiger_shp.exists():
        print(f"ERROR: TIGER shapefile missing: {tiger_shp}", file=sys.stderr)
        sys.exit(1)

    print(f"[prep] loading TIGER 2020 CA tracts from {tiger_shp}")
    tiger = load_tiger(str(tiger_shp))
    ca_geoids = set(tiger["GEOID"].values)
    print(f"[prep] TIGER: {len(ca_geoids)} CA tracts")

    for cat in args.categories:
        csv_name = CATS[cat]
        csv_path = root / csv_name
        if not csv_path.exists():
            print(f"[prep] skip {cat}: {csv_path} missing")
            continue

        print(f"[prep] {cat}: pivoting {csv_name}")
        pivot = reshape_category(str(csv_path), ca_geoids)
        T, N = pivot.shape
        print(f"[prep] {cat}: pivot shape T={T} N={N}")

        # Align metadata: keep tracts that are in pivot, in pivot's column order
        present_geoids = pivot.columns.tolist()
        meta = tiger.set_index("GEOID").reindex(present_geoids).reset_index()
        assert (meta["GEOID"].values == np.array(present_geoids)).all()

        # Save flow as Time-HD CSV (date, GEOID1, GEOID2, ...)
        flow_out = out / f"{cat}_flow.csv"
        pivot_save = pivot.reset_index()
        pivot_save["date"] = pivot_save["date"].dt.strftime("%Y-%m-%d")
        pivot_save.to_csv(flow_out, index=False)

        # Save meta CSV
        meta_out = out / f"{cat}_meta.csv"
        meta_save = pd.DataFrame({
            "GEOID": meta["GEOID"],
            "lat": meta["INTPTLAT"],
            "lng": meta["INTPTLON"],
            "county_fips": meta["COUNTYFP"],
            "area_sqkm": meta["area_sqkm"],
        })
        meta_save.to_csv(meta_out, index=False)

        # Save adjacency
        if not args.skip_adj:
            try:
                print(f"[prep] {cat}: building queen adjacency for N={N} (libpysal)…")
                # Need geopandas dataframe in the right row order
                meta_geo = tiger.set_index("GEOID").loc[present_geoids].reset_index()
                adj = queen_adj(meta_geo)
            except ImportError:
                print(f"[prep] {cat}: libpysal missing, falling back to shapely (slower)")
                meta_geo = tiger.set_index("GEOID").loc[present_geoids].reset_index()
                adj = queen_adj_fallback(meta_geo)
            adj_out = out / f"{cat}_adj.npy"
            np.save(adj_out, adj)
            print(f"[prep] {cat}: adj saved to {adj_out}, nz={int(adj.sum())} sparsity={adj.mean():.4f}")

        print(f"[prep] {cat}: wrote flow={flow_out.stat().st_size/1e6:.1f}MB meta={meta_out.stat().st_size/1e3:.1f}KB")
        print()


if __name__ == "__main__":
    main()
