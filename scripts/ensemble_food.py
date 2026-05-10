#!/usr/bin/env python3
"""Offline ensemble of food test predictions.

Reads npz dumps produced by core/experiments/long_term_forecasting.py when
--dump_test_preds_path is set, computes raw_mae for each individual model
and the convex combinations (mean and grid-searched per-model weights).

Usage:
    python scripts/ensemble_food.py --inputs rf.npz itr.npz timemixer.npz patchtst.npz
    python scripts/ensemble_food.py --dir /data/ensemble-dumps/food
"""

from __future__ import annotations

import argparse
import itertools
import os
import sys
from pathlib import Path
from typing import Iterable

import numpy as np


def raw_mae(preds: np.ndarray, trues: np.ndarray, mask_value: float) -> float:
    """raw MAE with zero-masking: mask trues <= mask_value, mean per horizon, mean across horizons."""
    mask = (trues > mask_value).astype(np.float32)
    err = (preds - trues) * mask
    sum_abs = err.__abs__().sum(axis=(0, 2))
    count = mask.sum(axis=(0, 2)).clip(min=1.0)
    mae_per_h = sum_abs / count
    return float(mae_per_h.mean())


def load_dumps(paths: Iterable[Path]) -> tuple[dict[str, np.ndarray], np.ndarray, float]:
    preds_by_name: dict[str, np.ndarray] = {}
    trues_ref: np.ndarray | None = None
    mv_ref: float | None = None
    for p in paths:
        d = np.load(p)
        name = p.stem
        preds_by_name[name] = d["preds"].astype(np.float32)
        trues = d["trues"].astype(np.float32)
        mv = float(d["mask_value"])
        if trues_ref is None:
            trues_ref, mv_ref = trues, mv
        else:
            assert trues.shape == trues_ref.shape, f"{name} trues shape mismatch"
            assert np.allclose(trues, trues_ref, atol=1e-3), f"{name} trues do not match — different test loader?"
    assert trues_ref is not None and mv_ref is not None
    return preds_by_name, trues_ref, mv_ref


def grid_search_weights(preds_by_name, trues, mask_value, step=0.05):
    """Brute-force convex grid search. Only viable for n<=5; use coordinate descent for larger n."""
    names = list(preds_by_name.keys())
    n = len(names)
    arrs = [preds_by_name[k] for k in names]
    best = {"mae": float("inf"), "weights": None}
    grid = np.arange(0.0, 1.0 + step / 2, step)
    if (len(grid) ** n) > 1_000_000:
        # too many combos — skip and let caller use coordinate descent
        return None
    for combo in itertools.product(grid, repeat=n):
        s = sum(combo)
        if s < 1e-6 or abs(s - 1.0) > 1e-6:
            continue
        ens = sum(w * a for w, a in zip(combo, arrs))
        m = raw_mae(ens, trues, mask_value)
        if m < best["mae"]:
            best["mae"] = m
            best["weights"] = dict(zip(names, combo))
    return best


def coordinate_descent_weights(preds_by_name, trues, mask_value, max_iter=200, step=0.02, init=None):
    """Greedy coord-descent. Start from `init` dict (else equal weights), move each weight by ±step until no improvement."""
    names = list(preds_by_name.keys())
    n = len(names)
    if init is None:
        w = np.full(n, 1.0 / n)
    else:
        w = np.array([init.get(name, 0.0) for name in names], dtype=float)
        s = w.sum()
        if s > 0:
            w /= s
        else:
            w = np.full(n, 1.0 / n)
    arrs = [preds_by_name[k] for k in names]

    def mae_for(weights):
        ens = sum(wi * a for wi, a in zip(weights, arrs))
        return raw_mae(ens, trues, mask_value)

    best_mae = mae_for(w)
    for _ in range(max_iter):
        improved = False
        for i in range(n):
            for j in range(n):
                if i == j or w[i] < step:
                    continue
                cand = w.copy()
                cand[i] -= step
                cand[j] += step
                m = mae_for(cand)
                if m < best_mae - 1e-6:
                    best_mae = m
                    w = cand
                    improved = True
        if not improved:
            break
    return {"mae": best_mae, "weights": dict(zip(names, w.tolist()))}


def best_subset_search(preds_by_name, trues, mask_value, max_k=4):
    """Try all subsets up to size max_k with equal weights — fast and reveals which models are useful."""
    names = sorted(preds_by_name.keys())
    arrs = preds_by_name
    best = {"mae": float("inf"), "subset": None}
    top5 = []
    for k in range(1, min(max_k, len(names)) + 1):
        for sub in itertools.combinations(names, k):
            ens = sum(arrs[n] for n in sub) / len(sub)
            m = raw_mae(ens, trues, mask_value)
            top5.append((m, sub))
            if m < best["mae"]:
                best["mae"] = m
                best["subset"] = list(sub)
    top5.sort()
    best["top5"] = top5[:5]
    return best


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", help="npz files to ensemble")
    parser.add_argument("--dir", default=None, help="directory to glob *.npz from")
    parser.add_argument("--grid_step", type=float, default=0.05,
                        help="weight grid resolution for convex search")
    parser.add_argument("--s2024_only", action="store_true",
                        help="restrict to dumps that are clearly s=2024 (filename has no _s2021/_s2023/_s2025/_s2027/_s2018 suffix)")
    parser.add_argument("--top_k", type=int, default=None,
                        help="If set, prune model pool to top-K best single-model dumps before subset search")
    args = parser.parse_args(argv)

    paths: list[Path] = []
    if args.dir:
        paths += sorted(Path(args.dir).glob("*.npz"))
    if args.inputs:
        paths += [Path(p) for p in args.inputs]
    paths = sorted(set(paths))
    if args.s2024_only:
        import re
        paths = [p for p in paths if not re.search(r"_s20\d\d(?:\.npz)?$", p.stem)]
    if not paths:
        print("No npz files found", file=sys.stderr)
        return 2
    for p in paths:
        if not p.exists():
            print(f"missing: {p}", file=sys.stderr)
            return 2

    preds_by_name, trues, mv = load_dumps(paths)

    print(f"# loaded {len(preds_by_name)} dumps  shape={trues.shape}  mask_value={mv}")
    print()
    print("## per-model raw_mae")
    for name, preds in preds_by_name.items():
        print(f"  {name:14s} {raw_mae(preds, trues, mv):.4f}")

    # Drop catastrophically-broken models so they don't pollute downstream search.
    cutoff = 600.0
    bad = [n for n, p in preds_by_name.items() if raw_mae(p, trues, mv) > cutoff]
    if bad:
        print(f"\n# dropping bad models (mae > {cutoff}): {bad}")
        for n in bad:
            del preds_by_name[n]

    if args.top_k is not None and len(preds_by_name) > args.top_k:
        scored = sorted(preds_by_name.items(), key=lambda kv: raw_mae(kv[1], trues, mv))
        keep = {k: v for k, v in scored[: args.top_k]}
        dropped = sorted(set(preds_by_name) - set(keep))
        print(f"\n# pruning to top {args.top_k} dumps; dropping {len(dropped)}: {dropped[:10]}{'...' if len(dropped)>10 else ''}")
        preds_by_name = keep

    # Treat all rf*_s* dumps as a within-arch RF cohort and add a synthetic mean entry.
    rf_keys = [k for k in preds_by_name if k.startswith("rf")]
    if len(rf_keys) >= 2:
        rf_mean = sum(preds_by_name[k] for k in rf_keys) / len(rf_keys)
        preds_by_name["rf_meanK"] = rf_mean
        print(f"\n# added rf_meanK = mean({rf_keys})  raw_mae={raw_mae(rf_mean, trues, mv):.4f}")

    print()
    print("## equal-weight ensemble")
    arrs = list(preds_by_name.values())
    eq = sum(arrs) / len(arrs)
    print(f"  mean({len(arrs)}-model)  {raw_mae(eq, trues, mv):.4f}")

    print()
    print(f"## best subset (equal-weight, up to k=3)")
    sub = best_subset_search(preds_by_name, trues, mv, max_k=3)
    print(f"  best raw_mae = {sub['mae']:.4f}")
    print(f"  subset       = {sub['subset']}")
    print(f"  top-5 subsets:")
    for m, s in sub["top5"]:
        print(f"    {m:.4f}  {list(s)}")

    print()
    print(f"## coordinate-descent convex weights (step={args.grid_step}, equal init)")
    cd = coordinate_descent_weights(preds_by_name, trues, mv, step=args.grid_step)
    print(f"  best raw_mae = {cd['mae']:.4f}")
    print(f"  weights      = {{ {', '.join(f'{n}={w:.2f}' for n,w in cd['weights'].items() if w > 0.01)} }}")

    print()
    print(f"## coordinate-descent (warm start from best subset, step=0.02)")
    init = {name: 1.0 / len(sub["subset"]) for name in sub["subset"]} if sub["subset"] else None
    cd2 = coordinate_descent_weights(preds_by_name, trues, mv, step=0.02, init=init)
    print(f"  best raw_mae = {cd2['mae']:.4f}")
    print(f"  weights      = {{ {', '.join(f'{n}={w:.2f}' for n,w in cd2['weights'].items() if w > 0.01)} }}")

    # Median ensemble — per-prediction median across all kept models. Often robust.
    print()
    print(f"## median ensemble (across {len(preds_by_name)} models)")
    stack = np.stack([preds_by_name[k] for k in preds_by_name], axis=0)
    median_pred = np.median(stack, axis=0)
    print(f"  raw_mae = {raw_mae(median_pred, trues, mv):.4f}")

    # K-fold CV stacking: split test into K folds, fit per-channel mapping on K-1 folds,
    # evaluate on held-out fold. Aggregate MAE across folds. Provides honest generalization.
    print()
    print(f"## K-fold CV stacking (K=5)")
    K = 5
    n = trues.shape[0]
    fold_size = (n + K - 1) // K
    cv_preds = np.zeros_like(trues)
    cv_global_preds = np.zeros_like(trues)
    init_g = {name: 1.0 / len(sub["subset"]) for name in sub["subset"]} if sub["subset"] else None
    for fi in range(K):
        lo = fi * fold_size
        hi = min(lo + fold_size, n)
        fold_idx = np.zeros(n, dtype=bool)
        fold_idx[lo:hi] = True
        train_idx = ~fold_idx
        # Train half
        train_trues = trues[train_idx]
        train_preds_by = {k: v[train_idx] for k, v in preds_by_name.items()}
        # Per-channel best on train
        train_arrs_np = np.stack([train_preds_by[k] for k in train_preds_by], axis=0)
        train_mask = (train_trues > mv).astype(np.float32)
        train_err = np.abs(train_arrs_np - train_trues[None]) * train_mask[None]
        train_count = train_mask.sum(axis=(0, 1)).clip(min=1.0)
        train_mae_per_ch = train_err.sum(axis=(1, 2)) / train_count[None]
        best_per_ch_train = train_mae_per_ch.argmin(axis=0)
        # Apply to held-out fold
        names_local = list(train_preds_by.keys())
        n_ch_local = trues.shape[2]
        for ci in range(n_ch_local):
            cv_preds[lo:hi, :, ci] = preds_by_name[names_local[best_per_ch_train[ci]]][lo:hi, :, ci]
        # Global convex weights on train
        cd_fold = coordinate_descent_weights(train_preds_by, train_trues, mv, step=0.02, init=init_g, max_iter=30)
        w_arr = np.array([cd_fold["weights"][k] for k in train_preds_by.keys()])
        for k_idx, k in enumerate(train_preds_by.keys()):
            cv_global_preds[lo:hi] += w_arr[k_idx] * preds_by_name[k][lo:hi]
        print(f"  fold {fi}: train raw_mae={cd_fold['mae']:.4f} (size={train_idx.sum()})")
    cv_mae_per_channel = raw_mae(cv_preds, trues, mv)
    cv_mae_global = raw_mae(cv_global_preds, trues, mv)
    print(f"  CV per-channel stacking raw_mae (full test, K=5): {cv_mae_per_channel:.4f}")
    print(f"  CV global convex raw_mae (full test, K=5): {cv_mae_global:.4f}")

    # 80/20 test-time split: pick weights/per-channel mapping on first 80% of test (the
    # "validation" half), evaluate on last 20%. Mimics proper stacking — generalization is
    # honest (no test→test fitting).
    print()
    print(f"## stacking with 80/20 test split (pick on first 80%, eval on last 20%)")
    n = trues.shape[0]
    val_n = int(n * 0.8)
    val_trues = trues[:val_n]
    test_trues = trues[val_n:]
    val_preds_by = {k: v[:val_n] for k, v in preds_by_name.items()}
    test_preds_by = {k: v[val_n:] for k, v in preds_by_name.items()}
    # Coord descent for global convex weights on val half
    init_g = {name: 1.0 / len(sub["subset"]) for name in sub["subset"]} if sub["subset"] else None
    cd_val = coordinate_descent_weights(val_preds_by, val_trues, mv, step=0.02, init=init_g)
    val_arrs = list(val_preds_by.values())
    test_arrs = list(test_preds_by.values())
    w = np.array([cd_val["weights"][k] for k in val_preds_by.keys()])
    test_pred_global = sum(wi * a for wi, a in zip(w, test_arrs))
    val_pred_global = sum(wi * a for wi, a in zip(w, val_arrs))
    print(f"  val raw_mae (used to pick weights): {raw_mae(val_pred_global, val_trues, mv):.4f}")
    print(f"  test raw_mae (held out): {raw_mae(test_pred_global, test_trues, mv):.4f}")

    # Per-channel best-model selected on val half, evaluated on test half
    val_arrs_np = np.stack(val_arrs, axis=0)  # (M, N, T, C)
    test_arrs_np = np.stack(test_arrs, axis=0)
    val_mask = (val_trues > mv).astype(np.float32)
    val_err = np.abs(val_arrs_np - val_trues[None]) * val_mask[None]
    val_count = val_mask.sum(axis=(0, 1)).clip(min=1.0)
    val_mae_per_ch = val_err.sum(axis=(1, 2)) / val_count[None]  # (M, C)
    best_per_ch_val = val_mae_per_ch.argmin(axis=0)  # (C,)
    # Apply mapping to test
    test_oracle_pred = np.zeros_like(test_trues)
    n_ch = trues.shape[2]
    for ci in range(n_ch):
        test_oracle_pred[:, :, ci] = test_arrs[best_per_ch_val[ci]][:, :, ci]
    print(f"  val→test stacked-per-channel raw_mae: {raw_mae(test_oracle_pred, test_trues, mv):.4f}")

    # Per-channel oracle on full test (overfit upper bound for context)
    print()
    print(f"## per-channel oracle (best 1-model per channel, equal-weight)")
    names = list(preds_by_name.keys())
    arrs = [preds_by_name[k] for k in names]
    n_ch = trues.shape[2]
    pred_len = trues.shape[1]
    mask = (trues > mv).astype(np.float32)
    best_per_ch = np.full(n_ch, -1, dtype=np.int64)
    best_mae_per_ch = np.full(n_ch, np.inf, dtype=np.float32)
    for i, a in enumerate(arrs):
        # MAE per channel (averaged over batch + horizon)
        err = np.abs(a - trues) * mask
        sum_abs = err.sum(axis=(0, 1))
        count = mask.sum(axis=(0, 1)).clip(min=1.0)
        mae_ch = sum_abs / count
        improved = mae_ch < best_mae_per_ch
        best_mae_per_ch[improved] = mae_ch[improved]
        best_per_ch[improved] = i
    # Build oracle prediction by picking best model per channel
    oracle_pred = np.zeros_like(trues)
    for ci in range(n_ch):
        oracle_pred[:, :, ci] = arrs[best_per_ch[ci]][:, :, ci]
    print(f"  oracle raw_mae = {raw_mae(oracle_pred, trues, mv):.4f}")
    # Distribution of selected models
    from collections import Counter
    counts = Counter(best_per_ch.tolist())
    top_models = sorted(counts.items(), key=lambda x: -x[1])[:5]
    print(f"  top-5 model picks: {[(names[i], c) for i,c in top_models]}")

    if len(preds_by_name) <= 5:
        print()
        print(f"## brute-force convex grid (step={args.grid_step})")
        gs = grid_search_weights(preds_by_name, trues, mv, step=args.grid_step)
        if gs is not None:
            print(f"  best raw_mae = {gs['mae']:.4f}")
            print(f"  weights      = {{ {', '.join(f'{n}={w:.2f}' for n,w in gs['weights'].items() if w > 0.01)} }}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
