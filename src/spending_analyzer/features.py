"""Cleaning and feature engineering."""

import warnings

import numpy as np
import pandas as pd

from .config import load_amount_binning


def normalize_merchant(s):
    return (
        s.fillna("")
        .astype(str)
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def _looks_day_first(s):
    parts = s.astype(str).str.extract(r"^\s*(\d{1,2})[/-](\d{1,2})[/-]\d{2,4}")
    first = pd.to_numeric(parts[0], errors="coerce")
    second = pd.to_numeric(parts[1], errors="coerce")
    return bool((first > 12).any() and not (second > 12).any())


def parse_timestamps(s):
    """Parse a timestamp column, picking day-first or month-first for the whole column.

    pandas infers one format from the first non-null value and applies it to
    everything, which is what we want, but it guesses wrong when the early rows
    are ambiguous (01-07-2019 in an otherwise day-first export). So try both and
    keep whichever explains more of the column.
    """
    if pd.api.types.is_datetime64_any_dtype(s):
        return s

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)  # the dayfirst warning is the point
        month_first = pd.to_datetime(s, errors="coerce")
        day_first = pd.to_datetime(s, errors="coerce", dayfirst=True)

    n_month, n_day = month_first.notna().sum(), day_first.notna().sum()
    if n_day != n_month:
        return day_first if n_day > n_month else month_first
    return day_first if _looks_day_first(s) else month_first


def assign_amount_bin(amount, binning=None):
    """Bucket amounts against fixed edges.

    amount_bin is a model input, so the edges have to be the same every run.
    Recomputing quantiles per upload moves a transaction between bins depending
    on what else was in the file, and gives NaN when the file is smaller than
    the bin count.
    """
    binning = binning or load_amount_binning()
    edges = [-np.inf] + list(binning.edges) + [np.inf]
    binned = pd.cut(amount, bins=edges, labels=False, include_lowest=True)
    return pd.to_numeric(binned, errors="coerce").fillna(binning.n_bins // 2).astype("int64")


def clean_transactions(df):
    """Coerce the canonical columns and drop rows we can't use.

    Returns (cleaned, dropped, reasons) where reasons counts bad timestamps and
    bad amounts separately.
    """
    out = df.copy()
    n_before = len(out)

    out["timestamp"] = parse_timestamps(out["timestamp"])
    out["amount"] = pd.to_numeric(out["amount"], errors="coerce")
    out["merchant"] = normalize_merchant(out["merchant"])

    reasons = {
        "bad_timestamp": int(out["timestamp"].isna().sum()),
        "bad_amount": int(out["amount"].isna().sum()),
    }
    out = out.dropna(subset=["timestamp", "amount"]).reset_index(drop=True)
    return out, n_before - len(out), reasons


def build_features(df, binning=None):
    """Build exactly the columns the categorization pipeline expects."""
    out, dropped, reasons = clean_transactions(df)
    if out.empty:
        for c in ["day_of_week", "hour", "log_amount", "amount_bin"]:
            out[c] = pd.Series(dtype="float64")
        return out, dropped, reasons

    out["day_of_week"] = out["timestamp"].dt.dayofweek
    out["hour"] = out["timestamp"].dt.hour
    out["log_amount"] = np.log1p(out["amount"].clip(lower=0))
    out["amount_bin"] = assign_amount_bin(out["amount"], binning)
    return out, dropped, reasons


def fit_amount_bin_edges(amount, n_bins):
    """Interior quantile cut points for a training amount distribution."""
    amount = pd.to_numeric(amount, errors="coerce").dropna()
    if amount.empty:
        raise ValueError("Cannot fit amount bins: no numeric amounts provided.")
    qs = np.linspace(0, 1, n_bins + 1)[1:-1]
    return [round(float(e), 4) for e in np.unique(np.quantile(amount, qs))]
