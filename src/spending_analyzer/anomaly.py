"""Isolation Forest anomaly detection over engineered features."""

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

MIN_ROWS = 10
RARE_MERCHANT_COUNT = 3
HIGH_AMOUNT_Q = 0.95
BUSY_DAY_Q = 0.90
NIGHT_START, NIGHT_END = 6, 23

DISPLAY_COLS = ["timestamp", "merchant", "amount", "anomaly_score", "anomaly_reason"]


class AnomalyResult:
    """Top anomalies for display, plus the totals behind them."""

    def __init__(self, anomalies, total_rows, total_anomalies, scores=None):
        self.anomalies = anomalies
        self.total_rows = total_rows
        self.total_anomalies = total_anomalies
        self.scores = pd.Series(dtype="float64") if scores is None else scores

    @property
    def is_truncated(self):
        return len(self.anomalies) < self.total_anomalies


def _feature_matrix(df):
    f = pd.DataFrame(index=df.index)

    f["hour"] = df["timestamp"].dt.hour
    f["day_of_week"] = df["timestamp"].dt.dayofweek
    f["day_of_month"] = df["timestamp"].dt.day
    f["is_weekend"] = (f["day_of_week"] >= 5).astype(int)

    f["amount"] = df["amount"]
    f["log_amount"] = np.log1p(df["amount"].clip(lower=0))

    counts = df["merchant"].map(df["merchant"].value_counts())
    f["merchant_freq"] = counts
    f["is_rare_merchant"] = (counts <= RARE_MERCHANT_COUNT).astype(int)

    if "pred_category" in df.columns:
        cat_median = df.groupby("pred_category")["amount"].transform("median")
        f["amount_vs_category_median"] = df["amount"] / (cat_median + 1)

    day = df["timestamp"].dt.normalize()
    f["daily_txn_count"] = day.map(day.value_counts())
    return f


def _reasons(df, f):
    """Readable labels for why a row stood out.

    Kept vectorised on purpose. Looping row by row and recomputing the quantiles
    inside the loop turns this quadratic (~16s on 20k rows).
    """
    high_amount = df["amount"].quantile(HIGH_AMOUNT_Q)
    busy_day = f["daily_txn_count"].quantile(BUSY_DAY_Q)

    flags = {
        "rare_merchant": f["is_rare_merchant"].astype(bool),
        "high_amount": df["amount"] > high_amount,
        "high_frequency_day": f["daily_txn_count"] > busy_day,
        "unusual_hour": (f["hour"] < NIGHT_START) | (f["hour"] > NIGHT_END),
    }
    parts = pd.DataFrame(
        {name: np.where(mask.to_numpy(), name + ",", "") for name, mask in flags.items()},
        index=df.index,
    )
    return parts.sum(axis=1).str.rstrip(",").replace("", "pattern_based")


def detect_multi_dimensional_anomalies(facts, contamination=0.05, max_results=50, random_state=42):
    """Score every transaction and return the worst `max_results` of them."""
    if facts is None or facts.empty or len(facts) < MIN_ROWS:
        rows = 0 if facts is None else len(facts)
        return AnomalyResult(pd.DataFrame(columns=DISPLAY_COLS), rows, 0)

    df = facts.copy()
    f = _feature_matrix(df)

    X = StandardScaler().fit_transform(f.fillna(0.0).to_numpy(dtype=float))
    forest = IsolationForest(
        contamination=contamination, random_state=random_state, n_estimators=100
    )
    labels = forest.fit_predict(X)

    # score_samples is negative and lower means more anomalous, so flip it to
    # keep "higher is worse" everywhere in the UI.
    df["anomaly_score"] = -forest.score_samples(X)
    df["is_anomaly"] = labels == -1
    df["anomaly_reason"] = _reasons(df, f)

    cols = list(DISPLAY_COLS)
    if "pred_category" in df.columns:
        cols.insert(3, "pred_category")

    hits = df.loc[df["is_anomaly"]].sort_values("anomaly_score", ascending=False)
    return AnomalyResult(
        anomalies=hits[cols].head(max_results).reset_index(drop=True),
        total_rows=len(df),
        total_anomalies=int(df["is_anomaly"].sum()),
        scores=df["anomaly_score"],
    )
