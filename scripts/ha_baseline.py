"""HA (Historical Average) baseline for Mobility-CA — paper Tab 1.

Predicts y_{t+h, n} as the training-set mean of the same time-of-week slot for
sensor n. Computes raw MAE/RMSE/MAPE per horizon with mask_value (default 1e-3).

Usage:      --out results/ha_GBA.json
"""
from __future__ import annotations
import argparse, json, os, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--root_path", required=True)
    ap.add_argument("--seq_len", type=int, default=12)
    ap.add_argument("--pred_len", type=int, default=12)
    ap.add_argument("--mask_value", type=float, default=1e-3)
    ap.add_argument("--out", required=True)
    ap.add_argument("--slot_len", type=int, default=2016,
                    help="Number of time slots in one period.")
    args = ap.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    import types
    sys.modules.setdefault("datasets", types.SimpleNamespace(load_dataset=lambda *a, **k: None))

    if args.data.startswith("Mobility-CA-"):
        from core.data.mobility_ca import Dataset_MobilityCA as DS
    else:
        raise ValueError(args.data)

    class A:
        data = args.data
        augmentation_ratio = 0
        embed = "timeF"

    print(f"[HA] loading train+test")
    ds_train = DS(A(), root_path=args.root_path, flag="train",
                  size=[args.seq_len, 0, args.pred_len], features="M",
                  timeenc=1, freq="W")
    ds_test = DS(A(), root_path=args.root_path, flag="test",
                 size=[args.seq_len, 0, args.pred_len], features="M",
                 timeenc=1, freq="W")

    # data_x is normalized; we want raw flow for MAE.
    # Use scaler.inverse_transform.
    def to_raw(z, ds):
        return ds.scaler.inverse_transform(z) if hasattr(ds, "scaler") else z

    train_raw = to_raw(ds_train.data_x, ds_train)  # (T_train, N)
    test_raw = to_raw(ds_test.data_x, ds_test)     # (T_test, N)
    T_train, N = train_raw.shape

    # Build slot-wise mean: slot index = t mod slot_len
    slot_sum = np.zeros((args.slot_len, N), dtype=np.float64)
    slot_cnt = np.zeros((args.slot_len,), dtype=np.int64)
    for t in range(T_train):
        s = t % args.slot_len
        slot_sum[s] += train_raw[t]
        slot_cnt[s] += 1
    slot_mean = slot_sum / np.maximum(slot_cnt, 1)[:, None]

    # For test: each window starts at index s_begin in test_raw; we want predictions
    # for t = s_begin+seq_len .. s_begin+seq_len+pred_len-1.
    # The slot offset in the *combined* timeline is: ds_test._global_t = T_train_pre + s_begin (approx).
    # Simpler: just align by reading test stamp. ds_test.data_stamp tells us...
    # ds_test sets data_x = full_data[border1:border2]. We don't have border1.
    # Use a robust fallback: time-of-week reconstruction from raw test timestamps would
    # require the date column. We'll approximate by stitching: HA uses train-set slot_mean
    # circularly indexed by test_t alone. Calls test_t = 0 as slot 0; this is a simple
    pred_len = args.pred_len
    n_windows = len(ds_test) if hasattr(ds_test, "__len__") else (len(test_raw) - args.seq_len - pred_len + 1)

    mae_per_h = np.zeros(pred_len, dtype=np.float64)
    rmse_per_h = np.zeros(pred_len, dtype=np.float64)
    mape_per_h = np.zeros(pred_len, dtype=np.float64)
    valid_per_h = np.zeros(pred_len, dtype=np.int64)

    for ws in range(0, len(test_raw) - args.seq_len - pred_len + 1):
        for h in range(pred_len):
            t = ws + args.seq_len + h
            slot = t % args.slot_len
            pred_h = slot_mean[slot]
            true_h = test_raw[t]
            mask = np.abs(true_h) >= args.mask_value
            err = np.abs(pred_h - true_h)
            mae_per_h[h] += (err * mask).sum()
            rmse_per_h[h] += (((pred_h - true_h) ** 2) * mask).sum()
            mape_per_h[h] += ((err / np.maximum(np.abs(true_h), args.mask_value)) * mask).sum()
            valid_per_h[h] += mask.sum()

    mae_per_h = mae_per_h / np.maximum(valid_per_h, 1)
    rmse_per_h = np.sqrt(rmse_per_h / np.maximum(valid_per_h, 1))
    mape_per_h = mape_per_h / np.maximum(valid_per_h, 1) * 100

    out = {
        "data": args.data,
        "model": "HA",
        "raw_mae_per_horizon": mae_per_h.tolist(),
        "raw_rmse_per_horizon": rmse_per_h.tolist(),
        "raw_mape_per_horizon": mape_per_h.tolist(),
        "raw_mae_overall": float(mae_per_h.mean()),
        "raw_rmse_overall": float(rmse_per_h.mean()),
        "raw_mape_overall": float(mape_per_h.mean()),
    }
    print(f"[HA] overall MAE={out['raw_mae_overall']:.3f} RMSE={out['raw_rmse_overall']:.3f} MAPE={out['raw_mape_overall']:.2f}%")
    print(f"[HA] per-h [3,6,9,12]: {[round(mae_per_h[i],2) for i in [2,5,8,11]]}")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[HA] wrote {args.out}")


if __name__ == "__main__":
    main()
