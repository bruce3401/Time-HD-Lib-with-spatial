"""Mobility-CA loader for IJGIS SpatialScale paper's case-study dataset.

Reads pre-processed Time-HD-format files from
  <root>/Mobility_CA/processed/<category>_flow.csv     (date, GEOID1, GEOID2, ...)
  <root>/Mobility_CA/processed/<category>_meta.csv     (GEOID, lat, lng, county_fips, area_sqkm)
  <root>/Mobility_CA/processed/<category>_adj.npy      (N, N) queen contiguity

Dataset key → category mapping:
  Mobility-CA-essential  → essential_retail
  Mobility-CA-food       → food_and_drink
  Mobility-CA-indoor     → indoor_discretionary
  Mobility-CA-outdoor    → outdoor_discretionary

Returns the standard `(seq_x, seq_y, seq_x_mark, seq_y_mark)` tuple and exposes
spatial metadata (lat/lng/adj/ids) as attributes for SpatialLCA's E1/E2.

Time resolution is weekly. seq_len/pred_len follow the framework convention.
"""
from __future__ import annotations
import os
import warnings
from typing import Optional

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from core.utils.timefeatures import time_features

warnings.filterwarnings("ignore")


CATEGORY_KEY = {
    "Mobility-CA-essential":  "essential_retail",
    "Mobility-CA-food":       "food_and_drink",
    "Mobility-CA-indoor":     "indoor_discretionary",
    "Mobility-CA-outdoor":    "outdoor_discretionary",
}


def _load_heat_weeks(noaa_csv: str) -> set:
    """Return a set of week_start datetime64[ns] from NOAA heat-week CSV."""
    if not os.path.exists(noaa_csv):
        return set()
    df = pd.read_csv(noaa_csv, parse_dates=["week_start"])
    return set(df["week_start"].dt.to_pydatetime())


class Dataset_MobilityCA(Dataset):
    """One NAICS-category × all weeks × all CA tracts.

    Optionally hold out NOAA heat-event weeks for the IJGIS case study:
    - flag = 'train' → all non-heat training weeks
    - flag = 'val'   → standard val portion
    - flag = 'test'  → standard test portion
    - flag = 'heat'  → ONLY heat-event weeks (case study eval set)
    """

    SPLIT_RATIOS = (0.7, 0.1, 0.2)

    def __init__(
        self,
        args,
        root_path: str,
        flag: str = "train",
        size=None,
        features: str = "M",
        data_path: Optional[str] = None,
        target: str = "OT",
        scale: bool = True,
        timeenc: int = 1,
        freq: str = "W",
        seasonal_patterns=None,
    ):
        self.args = args
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ("train", "val", "test", "heat")
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2, "heat": 3}.get(flag, 0)
        # NOAA heat-week hold-out CSV (optional; if present we offer a 'heat' split)
        self.heat_weeks_csv = getattr(args, "heat_weeks_csv", None)
        self.exclude_heat_from_train = bool(getattr(args, "exclude_heat_from_train", False))
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path

        key = getattr(args, "data", "Mobility-CA-outdoor")
        if key not in CATEGORY_KEY:
            raise ValueError(f"Unknown Mobility-CA subset {key!r}; expected one of {list(CATEGORY_KEY)}")
        self.subset = key
        self.category = CATEGORY_KEY[key]
        self.processed_dir = os.path.join(self.root_path, "processed")

        self.__read_data__()

    def __read_data__(self):
        flow_path = os.path.join(self.processed_dir, f"{self.category}_flow.csv")
        meta_path = os.path.join(self.processed_dir, f"{self.category}_meta.csv")
        adj_path = os.path.join(self.processed_dir, f"{self.category}_adj.npy")

        # Read flow with GEOID columns as strings (preserve leading zeros)
        df = pd.read_csv(flow_path, dtype=str)
        df["date"] = pd.to_datetime(df["date"])
        cols = [c for c in df.columns if c != "date"]
        df[cols] = df[cols].astype(np.float32)
        df = df.sort_values("date").reset_index(drop=True)

        flow = df[cols].to_numpy(dtype=np.float32)  # (T, N)
        stamp = df["date"].to_numpy(dtype="datetime64[ns]")
        T, N = flow.shape

        meta = pd.read_csv(meta_path, dtype={"GEOID": str, "county_fips": str})
        # Reorder meta to match flow column order
        meta = meta.set_index("GEOID").reindex(cols).reset_index()
        if meta.isna().any().any():
            raise ValueError("meta has NaNs after reindex — flow GEOIDs and meta GEOIDs misaligned")

        adj = np.load(adj_path).astype(np.float32)
        assert adj.shape == (N, N), f"adj {adj.shape} != flow N={N}"

        # Splits
        r_train, r_val, _ = self.SPLIT_RATIOS
        n_total = T
        n_train = int(n_total * r_train)
        n_val = int(n_total * r_val)
        border1s = [0, n_train - self.seq_len, n_total - (n_total - n_train - n_val) - self.seq_len]
        border2s = [n_train, n_train + n_val, n_total]

        # Identify heat-event week indices (set of T indices) if NOAA file provided
        heat_idx = set()
        if self.heat_weeks_csv:
            heat_set = _load_heat_weeks(self.heat_weeks_csv)
            for ti in range(T):
                ts_py = pd.Timestamp(stamp[ti]).to_pydatetime().replace(hour=0, minute=0, second=0, microsecond=0)
                # week_start in NOAA is week-of-Sunday. Mobility-CA dates are weekly.
                # Match by year+week-of-year.
                for hw in heat_set:
                    if abs((hw - ts_py).days) <= 6:
                        heat_idx.add(ti)
                        break

        if self.flag == "heat":
            # Build a "heat" split that contains only heat-event weeks (for case study eval).
            if not heat_idx:
                raise RuntimeError("flag='heat' requested but no heat weeks matched. Check heat_weeks_csv.")
            # Use whole timeline; mask out non-heat windows by setting border1/border2 wide
            # but actual indexing happens via custom __getitem__ list.
            self._heat_indices = sorted(heat_idx)
            b1, b2 = 0, n_total  # use full data for inverse_transform reference
        else:
            b1, b2 = border1s[self.set_type], border2s[self.set_type]
            # Optionally exclude heat weeks from training set (data leakage prevention for case study)
            if self.flag == "train" and self.exclude_heat_from_train and heat_idx:
                # We can't easily remove rows from data_x without breaking sliding-window indexing.
                # Instead, store a mask of valid window-start indices.
                self._exclude_heat_window_starts = set()
                for ti in heat_idx:
                    if 0 <= ti - b1 < (b2 - b1) - self.seq_len - self.pred_len + 1:
                        # If a window starting at ws would touch heat at ti, exclude ws
                        ws_min = max(0, ti - b1 - self.seq_len - self.pred_len + 1)
                        ws_max = min((b2 - b1) - self.seq_len - self.pred_len, ti - b1)
                        for ws in range(ws_min, ws_max + 1):
                            self._exclude_heat_window_starts.add(ws)

        scaler = StandardScaler()
        if self.scale:
            scaler.fit(flow[: border2s[0]])
            data = scaler.transform(flow).astype(np.float32)
        else:
            data = flow

        df_stamp = pd.DataFrame({"date": pd.to_datetime(stamp[b1:b2])})
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["weekofyear"] = df_stamp.date.dt.isocalendar().week.astype(int)
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            data_stamp = df_stamp.drop(columns=["date"]).to_numpy(dtype=np.float32)
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values), freq=self.freq).T.astype(np.float32)

        self.scaler = scaler
        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp

        # Spatial attrs
        self.lat = meta["lat"].to_numpy(dtype=np.float32)
        self.lng = meta["lng"].to_numpy(dtype=np.float32)
        self.adj = adj
        self.ids = meta["GEOID"].to_numpy()
        self.county_fips = meta["county_fips"].to_numpy()
        self.n_nodes = N

    def __getitem__(self, index):
        if self.flag == "heat" and hasattr(self, "_heat_indices"):
            # Each window's last input timestep falls on a heat-event week
            t_heat = self._heat_indices[index]
            s_begin = max(0, t_heat - self.seq_len + 1)
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
            if r_end > len(self.data_x):
                # Pad last by clamp
                r_end = len(self.data_x); r_begin = r_end - self.pred_len - self.label_len
        else:
            # Optionally skip heat weeks from training
            if self.flag == "train" and getattr(self, "_exclude_heat_window_starts", None):
                # Remap index to skip excluded window starts
                pos = 0
                for ws in range(len(self.data_x) - self.seq_len - self.pred_len + 1):
                    if ws in self._exclude_heat_window_starts:
                        continue
                    if pos == index:
                        index = ws
                        break
                    pos += 1
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_y[r_begin:r_end]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.flag == "heat" and hasattr(self, "_heat_indices"):
            return len(self._heat_indices)
        n = len(self.data_x) - self.seq_len - self.pred_len + 1
        if self.flag == "train" and getattr(self, "_exclude_heat_window_starts", None):
            n -= len(self._exclude_heat_window_starts)
        return max(0, n)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
