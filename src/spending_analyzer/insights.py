"""Spending insights: recurring charges, outliers, patterns, projections."""

import numpy as np
import pandas as pd

from .features import parse_timestamps

RECURRING_COLS = [
    "merchant", "count", "median_interval_days", "amount_cv", "last_seen", "next_estimated",
]

MONTHLY_RANGE = (26, 33)
MIN_OCCURRENCES = 3
MAX_AMOUNT_CV = 0.2


def make_facts(df):
    """Add the date/week/month keys the aggregations group on."""
    facts = df.copy()
    facts["timestamp"] = parse_timestamps(facts["timestamp"])
    facts["amount"] = pd.to_numeric(facts["amount"], errors="coerce")
    facts = facts.dropna(subset=["timestamp", "amount"]).reset_index(drop=True)
    if facts.empty:
        for c in ["date", "week", "month"]:
            facts[c] = pd.Series(dtype="object")
        return facts
    facts["date"] = facts["timestamp"].dt.date
    facts["week"] = facts["timestamp"].dt.to_period("W").astype(str)
    facts["month"] = facts["timestamp"].dt.to_period("M").astype(str)
    return facts


def recurring_detection(facts):
    """Merchants billed roughly monthly for roughly the same amount.

    Everything here has to be keyed on merchant. Aggregating the amounts into a
    positionally indexed frame and joining merchant-indexed intervals onto it
    just produces NaN intervals and the filter below never matches anything.
    """
    empty = pd.DataFrame(columns=RECURRING_COLS)
    if facts is None or facts.empty or "merchant" not in facts.columns:
        return empty

    df = facts.dropna(subset=["timestamp", "amount"]).sort_values("timestamp")
    if df.empty:
        return empty

    g = df.groupby("merchant", sort=False)
    stats = g.agg(
        count=("amount", "size"),
        mean_amount=("amount", "mean"),
        std_amount=("amount", "std"),
        last_seen=("timestamp", "max"),
    )

    by_merchant = df.sort_values(["merchant", "timestamp"])
    gaps = by_merchant.groupby("merchant", sort=False)["timestamp"].diff()
    gaps = (gaps.dt.total_seconds() / 86400.0).groupby(by_merchant["merchant"]).median()
    stats["median_interval_days"] = gaps.reindex(stats.index)
    stats["amount_cv"] = (
        (stats["std_amount"] / stats["mean_amount"]).replace([np.inf, -np.inf], np.nan)
    )

    lo, hi = MONTHLY_RANGE
    keep = (
        (stats["count"] >= MIN_OCCURRENCES)
        & stats["median_interval_days"].between(lo, hi)
        & (stats["amount_cv"].fillna(0.0) < MAX_AMOUNT_CV)
    )

    rec = stats.loc[keep].reset_index()
    if rec.empty:
        return empty
    rec["next_estimated"] = rec["last_seen"] + pd.to_timedelta(rec["median_interval_days"], unit="D")
    return rec[RECURRING_COLS].sort_values(["median_interval_days", "merchant"]).reset_index(drop=True)


def anomaly_high_spend(facts, top_n=5):
    """Transactions furthest above the mean of their own category."""
    cols = ["timestamp", "merchant", "amount", "pred_category", "z"]
    if facts is None or facts.empty or "pred_category" not in facts.columns:
        return pd.DataFrame(columns=cols)

    df = facts.copy()
    by_cat = df.groupby("pred_category")["amount"]
    sigma = by_cat.transform("std").replace(0, np.nan)
    df["z"] = (df["amount"] - by_cat.transform("mean")) / sigma
    out = df.dropna(subset=["z"]).sort_values("z", ascending=False)
    return out[cols].head(top_n).reset_index(drop=True)


def analyze_spending_patterns(facts):
    """Behavioural summary. Works on a copy, the caller's frame is left alone."""
    if facts is None or facts.empty:
        return {}

    df = facts.copy()
    patterns = {}

    weekend = df["timestamp"].dt.dayofweek >= 5
    wknd_avg = df.loc[weekend, "amount"].mean()
    week_avg = df.loc[~weekend, "amount"].mean()
    if pd.notna(wknd_avg) and pd.notna(week_avg) and week_avg > 0:
        patterns["weekend_vs_weekday_ratio"] = float(wknd_avg / week_avg)

    period = pd.cut(
        df["timestamp"].dt.hour,
        bins=[-1, 5, 11, 17, 23],
        labels=["Night", "Morning", "Afternoon", "Evening"],
    )
    patterns["time_distribution"] = (
        df.assign(time_period=period)
        .groupby("time_period", observed=False)["amount"]
        .agg(["sum", "count"])
        .to_dict()
    )

    ordered = df.sort_values("timestamp")
    week_num = (ordered["timestamp"] - ordered["timestamp"].min()).dt.days // 7
    weekly = ordered.groupby(week_num)["amount"].sum()
    if len(weekly) >= 6:
        recent, baseline = weekly.iloc[-3:].mean(), weekly.iloc[:3].mean()
        if recent > baseline * 1.05:
            patterns["spending_trend"] = "increasing"
        elif recent < baseline * 0.95:
            patterns["spending_trend"] = "decreasing"
        else:
            patterns["spending_trend"] = "stable"

    shares = df["merchant"].value_counts(normalize=True)
    shares = shares[shares > 0]
    patterns["merchant_diversity_score"] = float(-(shares * np.log(shares)).sum())
    patterns["dow_avg_amount"] = df.groupby(df["timestamp"].dt.day_name())["amount"].mean().to_dict()
    return patterns


def generate_predictive_insights(facts, min_rows=30):
    """Rough forward look.

    "Now" is the newest row in the file, not the wall clock. Scoring a 2019
    export against today just marks every merchant overdue for a visit.
    """
    if facts is None or facts.empty or len(facts) < min_rows:
        return {}

    df = facts.sort_values("timestamp").copy()
    as_of = df["timestamp"].max()
    insights = {"as_of": as_of}

    monthly = df.groupby(df["timestamp"].dt.to_period("M"))["amount"].sum()
    if len(monthly) >= 3:
        insights["projected_next_month"] = float(monthly.iloc[-3:].mean())

        if "pred_category" in df.columns:
            per_cat = df.groupby([df["timestamp"].dt.to_period("M"), "pred_category"])["amount"].sum()
            growing = []
            for cat, series in per_cat.groupby(level="pred_category"):
                values = series.droplevel("pred_category")
                if len(values) >= 3 and values.iloc[-1] > values.iloc[-3]:
                    growing.append(cat)
            insights["growing_categories"] = sorted(growing)

    revisit = []
    for merchant in df["merchant"].value_counts().head(10).index:
        visits = df.loc[df["merchant"] == merchant, "timestamp"]
        if len(visits) < 3:
            continue
        avg_gap = visits.diff().dropna().dt.total_seconds().mean() / 86400.0
        if not np.isfinite(avg_gap) or avg_gap <= 0:
            continue
        since = (as_of - visits.max()).total_seconds() / 86400.0
        if since >= avg_gap * 0.8:
            revisit.append({
                "merchant": merchant,
                "avg_interval_days": float(avg_gap),
                "days_since_last": float(since),
            })
    revisit.sort(key=lambda r: r["days_since_last"] - r["avg_interval_days"], reverse=True)
    insights["merchants_due_for_revisit"] = revisit[:5]
    return insights
