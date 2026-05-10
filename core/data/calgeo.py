"""CalGeo-Bench loader — unified Dataset class for the 3 new CalGeo domains.

Reads pre-processed Time-HD-format triplets from
  <root>/CalGeo/<domain>/<channel>_flow.csv     (date, site_1, site_2, ...)
  <root>/CalGeo/<domain>/<channel>_meta.csv     (site_id, lat, lng, county_fips)
  <root>/CalGeo/<domain>/<channel>_adj.npy      (N, N) adjacency

Dataset key → (domain, channel) mapping:
  CalGeo-AirQuality-pm25     → AirQuality, pm25
  CalGeo-AirQuality-ozone    → AirQuality, ozone
  CalGeo-Solar-ghi           → Solar,      ghi
  CalGeo-Weather-tmax        → Weather,    tmax
  CalGeo-Weather-tmin        → Weather,    tmin
  CalGeo-Weather-prcp        → Weather,    prcp

(Mobility-CA is loaded by core/data/mobility_ca.py — kept separate because it
exposes additional NOAA heat-event hold-out logic for the case study.)
"""
from __future__ import annotations
import os
import warnings
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

from core.utils.timefeatures import time_features

warnings.filterwarnings("ignore")


CALGEO_KEY: dict = {
    # AirQuality
    "CalGeo-AirQuality-pm25":  ("AirQuality", "pm25"),
    "CalGeo-AirQuality-ozone": ("AirQuality", "ozone"),
    # Solar
    "CalGeo-Solar-ghi":        ("Solar", "ghi"),
    # Weather
    "CalGeo-Weather-tmax":     ("Weather", "tmax"),
    "CalGeo-Weather-tmin":     ("Weather", "tmin"),
    "CalGeo-Weather-prcp":     ("Weather", "prcp"),
}


class Dataset_CalGeo(Dataset):
    """One CalGeo domain × channel × all daily timestamps × all CA sites.

    Standard chronological 60/20/20 train/val/test split.
    """

    SPLIT_RATIOS = (0.6, 0.2, 0.2)

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
        freq: str = "D",
        seasonal_patterns=None,
    ):
        self.args = args
        self.seq_len = size[0]
        self.label_len = size[1]
        self.pred_len = size[2]
        assert flag in ("train", "val", "test")
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.timeenc = timeenc
        self.freq = freq
        self.root_path = root_path

        key = getattr(args, "data", "CalGeo-AirQuality-pm25")
        if key not in CALGEO_KEY:
            raise ValueError(f"Unknown CalGeo subset {key!r}; expected one of {list(CALGEO_KEY)}")
        self.subset = key
        self.domain, self.channel = CALGEO_KEY[key]

        self.__read_data__()

    def __read_data__(self):
        domain_dir = os.path.join(self.root_path, "CalGeo", self.domain)
        flow_path = os.path.join(domain_dir, f"{self.channel}_flow.csv")
        meta_path = os.path.join(domain_dir, f"{self.channel}_meta.csv")
        adj_path  = os.path.join(domain_dir, f"{self.channel}_adj.npy")

        df = pd.read_csv(flow_path, dtype=str)
        df["date"] = pd.to_datetime(df["date"])
        cols = [c for c in df.columns if c != "date"]
        df[cols] = df[cols].astype(np.float32)
        df = df.sort_values("date").reset_index(drop=True)

        flow = df[cols].to_numpy(dtype=np.float32)  # (T, N)
        stamp = df["date"].to_numpy(dtype="datetime64[ns]")
        T, N = flow.shape

        meta = pd.read_csv(meta_path, dtype={"site_id": str, "county_fips": str})
        meta = meta.set_index("site_id").reindex(cols).reset_index()
        # county_fips may legitimately be empty/NaN for some domains (e.g. NREL grid points
        # or NOAA stations that don't carry FIPS) — only validate the load-bearing fields.
        required = ["site_id", "lat", "lng"]
        if meta[required].isna().any().any():
            raise ValueError(
                f"meta has NaN in {required} after reindex — "
                f"flow cols sample: {cols[:3]}…, meta cols: {meta.columns.tolist()}")
        meta["county_fips"] = meta.get("county_fips", "").fillna("").astype(str)

        adj = np.load(adj_path).astype(np.float32)
        assert adj.shape == (N, N), f"adj {adj.shape} != flow N={N}"

        # Splits (chronological)
        r_train, r_val, _ = self.SPLIT_RATIOS
        n_total = T
        n_train = int(n_total * r_train)
        n_val = int(n_total * r_val)
        border1s = [0, n_train - self.seq_len, n_total - (n_total - n_train - n_val) - self.seq_len]
        border2s = [n_train, n_train + n_val, n_total]
        b1, b2 = border1s[self.set_type], border2s[self.set_type]

        scaler = StandardScaler()
        if self.scale:
            scaler.fit(flow[: border2s[0]])
            data = scaler.transform(flow).astype(np.float32)
        else:
            data = flow

        df_stamp = pd.DataFrame({"date": pd.to_datetime(stamp[b1:b2])})
        if self.timeenc == 0:
            df_stamp["month"] = df_stamp.date.dt.month
            df_stamp["day"] = df_stamp.date.dt.day
            df_stamp["weekday"] = df_stamp.date.dt.weekday
            data_stamp = df_stamp.drop(columns=["date"]).to_numpy(dtype=np.float32)
        else:
            data_stamp = time_features(pd.to_datetime(df_stamp["date"].values),
                                        freq=self.freq).T.astype(np.float32)

        self.scaler = scaler
        self.data_x = data[b1:b2]
        self.data_y = data[b1:b2]
        self.data_stamp = data_stamp

        # Spatial attrs (consumed by SpatialLCA / RegionFormer's set_spatial_metadata)
        self.lat = meta["lat"].to_numpy(dtype=np.float32)
        self.lng = meta["lng"].to_numpy(dtype=np.float32)
        self.adj = adj
        self.ids = meta["site_id"].to_numpy()
        self.county_fips = meta["county_fips"].fillna("").to_numpy()
        self.n_nodes = N

    def __getitem__(self, index):
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
        return max(0, len(self.data_x) - self.seq_len - self.pred_len + 1)

    def inverse_transform(self, data):
        return self.scaler.inverse_transform(data)
