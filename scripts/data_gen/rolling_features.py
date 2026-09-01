#!/usr/bin/env python3
"""
Rolling-window feature computation for UPI fraud data.

Computes per-entity (merchant_id, device_fingerprint, upi_handle) rolling
counts and amount sums over configurable time windows.

IMPORTANT: all windows are strictly BACKWARD-LOOKING from the current row's
timestamp — the current transaction is EXCLUDED from its own window to
prevent data leakage.

Approach: sort by timestamp, then for each entity use a pointer / binary
search approach (O(N log N) total) rather than naïve O(N²) loop.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Core rolling engine
# ---------------------------------------------------------------------------

def _rolling_counts_for_entity(
    ts_arr: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    """
    Given a sorted 1-D array of unix timestamps (seconds) for ONE entity,
    return an array of the same length where each element is the number of
    prior transactions within `window_seconds` BEFORE (not including) the
    current transaction.

    Uses a two-pointer sweep: O(N) per entity.
    """
    n = len(ts_arr)
    counts = np.empty(n, dtype=np.int32)
    left = 0
    for i in range(n):
        cutoff = ts_arr[i] - window_seconds
        # advance left pointer until ts_arr[left] > cutoff
        while left < i and ts_arr[left] <= cutoff:
            left += 1
        # elements in [left, i) are within the window but BEFORE position i
        counts[i] = i - left
    return counts


def _rolling_amount_sum_for_entity(
    ts_arr: np.ndarray,
    amt_arr: np.ndarray,
    window_seconds: int,
) -> np.ndarray:
    """Same logic but returns sum of amount_paise in the window."""
    n = len(ts_arr)
    sums = np.empty(n, dtype=np.float64)
    window_sum = 0.0
    left = 0
    for i in range(n):
        cutoff = ts_arr[i] - window_seconds
        while left < i and ts_arr[left] <= cutoff:
            window_sum -= amt_arr[left]
            left += 1
        sums[i] = window_sum
        window_sum += amt_arr[i]
    return sums


def compute_rolling_features(
    df: pd.DataFrame,
    entity_cols: list[str] | None = None,
    windows_min: list[int] | None = None,
    include_amount_sum: bool = True,
) -> pd.DataFrame:
    """
    Compute rolling-window features for a UPI transaction dataframe.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain columns: timestamp (ISO string or datetime), amount_paise,
        and each column in `entity_cols`.
    entity_cols : list of str
        Columns to group by when computing windows.
        Default: ["merchant_id", "device_fingerprint", "upi_handle"].
    windows_min : list of int
        Window sizes in minutes.
        Default: [5, 60]  → _5min and _1hr suffixes kept for model compatibility.
    include_amount_sum : bool
        Whether to also emit `{col}_amount_sum_{N}min` columns.

    Returns
    -------
    pd.DataFrame
        Original df with new columns appended.  The df is returned sorted by
        timestamp and has a fresh integer index.
    """
    if entity_cols is None:
        entity_cols = ["merchant_id", "device_fingerprint", "upi_handle"]
    if windows_min is None:
        windows_min = [5, 60]

    # Ensure sorted by timestamp (required for two-pointer to be correct)
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Unix seconds array (shared)
    ts_unix = df["timestamp"].astype(np.int64) // 10**9  # nanoseconds → seconds
    amt = df["amount_paise"].to_numpy(dtype=np.float64)

    # Build output arrays: initialize to 0
    out_count: dict[str, np.ndarray] = {}
    out_amount: dict[str, np.ndarray] = {}
    for col in entity_cols:
        for wmin in windows_min:
            count_key = f"{col}_txn_count_{wmin}min"
            out_count[count_key] = np.zeros(len(df), dtype=np.int32)
            if include_amount_sum:
                amt_key = f"{col}_amount_sum_{wmin}min"
                out_amount[amt_key] = np.zeros(len(df), dtype=np.float64)

    # Process each entity column
    for col in entity_cols:
        groups = df.groupby(col, sort=False).groups  # dict: value → Index
        for entity_val, idx in groups.items():
            # idx is the integer positions in the sorted df
            pos = idx.to_numpy()
            # ts and amounts for this entity, in the order they appear in sorted df
            # pos is already in ascending timestamp order because df is sorted
            ts_entity = ts_unix.to_numpy()[pos]
            amt_entity = amt[pos]

            for wmin in windows_min:
                window_sec = wmin * 60
                counts = _rolling_counts_for_entity(ts_entity, window_sec)
                count_key = f"{col}_txn_count_{wmin}min"
                out_count[count_key][pos] = counts

                if include_amount_sum:
                    sums = _rolling_amount_sum_for_entity(ts_entity, amt_entity, window_sec)
                    amt_key = f"{col}_amount_sum_{wmin}min"
                    out_amount[amt_key][pos] = sums

    # Drop placeholder columns that may already exist in df
    all_new_cols = list(out_count.keys()) + list(out_amount.keys())
    df = df.drop(columns=[c for c in all_new_cols if c in df.columns], errors="ignore")

    # Assign new columns
    for key, arr in out_count.items():
        df[key] = arr
    for key, arr in out_amount.items():
        df[key] = arr

    return df


# ---------------------------------------------------------------------------
# Leakage sanity check
# ---------------------------------------------------------------------------

def verify_no_leakage(df: pd.DataFrame, entity_col: str, window_min: int) -> bool:
    """
    Spot-check that no row's rolling count exceeds the number of prior rows
    for that entity within the window.  Returns True if clean.
    """
    count_col = f"{entity_col}_txn_count_{window_min}min"
    if count_col not in df.columns:
        return True

    df = df.copy()
    df["_ts"] = pd.to_datetime(df["timestamp"], utc=True)
    df = df.sort_values("_ts").reset_index(drop=True)
    window_sec = window_min * 60

    problems = []
    for entity_val, grp in df.groupby(entity_col, sort=False):
        for i, (_, row) in enumerate(grp.iterrows()):
            expected = sum(
                1 for j, (_, prev) in enumerate(grp.iterrows())
                if j < i and (row["_ts"] - prev["_ts"]).total_seconds() <= window_sec
            )
            if row[count_col] != expected:
                problems.append((entity_val, i, expected, row[count_col]))
        if len(problems) > 5:
            break  # enough evidence

    if problems:
        print(f"LEAKAGE DETECTED in {count_col}:")
        for p in problems[:5]:
            print(f"  entity={p[0]} row={p[1]} expected={p[2]} got={p[3]}")
        return False
    return True
