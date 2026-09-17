"""Unsupervised risk scoring plus the rules that explain an alert."""

import numpy as np
import pandas as pd

from .features import normalize_merchant, parse_timestamps

EARTH_RADIUS_KM = 6371.0088
ROLL_N = 20
BASELINE_N = 200
BASELINE_MIN = 50
CONC_ALPHA = 0.3

RULE_NAMES = ["high_z", "rapid_swipes", "repeat_small", "geo_jump", "velocity_spike"]

# Source header -> risk feature name. The canonical names map to themselves so a
# file that already went through align_to_canonical still works here.
SOURCE_COLS = {
    "trans_date_trans_time": "timestamp",
    "timestamp": "timestamp",
    "merchant": "merchant",
    "category": "category",
    "amt": "amount",
    "amount": "amount",
    "city": "city",
    "state": "state",
    "cc_num": "cc_num",
    "unix_time": "unix_time",
    "lat": "cust_lat",
    "long": "cust_long",
    "merch_lat": "merch_lat",
    "merch_long": "merch_long",
}


class RiskThresholds:
    def __init__(self, percentile=99.0, require_two_rules=True, z_cut=4.0,
                 geo_km=1000.0, velocity_multiplier=4.5):
        self.percentile = percentile
        self.require_two_rules = require_two_rules
        self.z_cut = z_cut
        self.geo_km = geo_km
        self.velocity_multiplier = velocity_multiplier


def haversine_km(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    return EARTH_RADIUS_KM * 2 * np.arcsin(np.sqrt(a))


def _add_roll(g, n=ROLL_N):
    g = g.sort_values("timestamp").copy()
    g["prev_txn_delta_min"] = g["timestamp"].diff().dt.total_seconds().div(60).fillna(1e6)
    roll = g["amount"].rolling(n, min_periods=1)
    g["roll_cnt_N"] = roll.count()
    g["roll_sum_N"] = roll.sum()
    g["roll_max_N"] = roll.max()
    stable = g["amount"].rolling(n, min_periods=5)
    g["roll_mean_N"] = stable.mean()
    g["roll_std_N"] = stable.std()
    g["amount_z"] = ((g["amount"] - g["roll_mean_N"]) / g["roll_std_N"].replace(0, np.nan)).fillna(0.0)
    return g


def _add_merchant_feats(g, alpha=CONC_ALPHA):
    """Running share of the most-used merchant, plus the two burst flags.

    Only the merchant being seen gets decayed here. That is how the shipped risk
    model was trained, so leave it alone unless you retrain.
    """
    g = g.sort_values("timestamp").copy()
    total = 0.0
    counts = {}
    shares = []
    for m in g["merchant"].to_numpy():
        total = total * (1 - alpha) + 1.0
        counts[m] = counts.get(m, 0.0) * (1 - alpha) + 1.0
        shares.append(max(counts.values()) / max(total, 1e-6))
    g["merchant_conc_ewm"] = shares
    g["repeat_small_charge"] = ((g["amount"] < 5) & (g["prev_txn_delta_min"] < 10)).astype(int)
    g["rapid_swipes"] = (g["prev_txn_delta_min"] < 2).astype(int)
    return g


def build_risk_features(df):
    """Behavioural features from either the raw Kaggle columns or our own."""
    present = {src: dst for src, dst in SOURCE_COLS.items() if src in df.columns}
    x = df[list(present)].rename(columns=present).copy()
    x = x.loc[:, ~x.columns.duplicated()]

    missing = [c for c in ["timestamp", "amount"] if c not in x.columns]
    if missing:
        accepted = sorted(k for k, v in SOURCE_COLS.items() if v in missing)
        raise ValueError(f"Risk scoring needs {missing}; none of {accepted} were found.")
    if "merchant" not in x.columns:
        x["merchant"] = ""

    x["timestamp"] = parse_timestamps(x["timestamp"])
    for c in ["amount", "unix_time", "cust_lat", "cust_long", "merch_lat", "merch_long"]:
        if c in x.columns:
            x[c] = pd.to_numeric(x[c], errors="coerce")
    for c in ["merchant", "city", "state"]:
        if c in x.columns:
            x[c] = normalize_merchant(x[c])

    x = x.dropna(subset=["timestamp", "amount"]).reset_index(drop=True)
    if x.empty:
        raise ValueError("No rows left with a usable timestamp and amount.")

    has_card = "cc_num" in x.columns and x["cc_num"].notna().any()
    x = x.sort_values(["cc_num", "timestamp"] if has_card else ["timestamp"])
    if has_card:
        x = x.groupby("cc_num", group_keys=False)[x.columns].apply(_add_roll)
        x = x.groupby("cc_num", group_keys=False)[x.columns].apply(_add_merchant_feats)
    else:
        x = _add_merchant_feats(_add_roll(x))

    x["hour"] = x["timestamp"].dt.hour
    x["dayofweek"] = x["timestamp"].dt.dayofweek
    x["is_weekend"] = (x["dayofweek"] >= 5).astype(int)

    geo = ["cust_lat", "cust_long", "merch_lat", "merch_long"]
    x["geo_dist_km"] = 0.0
    if all(c in x.columns for c in geo):
        ok = x[geo].notna().all(axis=1)
        x["geo_missing"] = (~ok).astype(int)
        if ok.any():
            x.loc[ok, "geo_dist_km"] = haversine_km(*(x.loc[ok, c].astype(float) for c in geo))
    else:
        x["geo_missing"] = 1

    return x.reset_index(drop=True)


def score_risk(df, bundle):
    """Rescale the detector's decision function to 0-1, higher being riskier.

    Both ends have to come from the same (negated) series, otherwise the score
    is just offset and runs past 1.
    """
    X = pd.DataFrame(
        {c: df[c] if c in df.columns else 0.0 for c in bundle.feature_cols},
        index=df.index,
    ).astype(float).fillna(0.0)

    scores = -np.asarray(bundle.pipeline.decision_function(X), dtype=float)
    spread = scores.max() - scores.min()
    if spread <= 0:
        return pd.Series(np.zeros(len(df)), index=df.index, name="risk_score")
    return pd.Series((scores - scores.min()) / spread, index=df.index, name="risk_score")


def apply_rules(df, thresholds):
    """Run the rule set and decide which rows become alerts."""
    out = df.copy()

    # min_periods leaves the warm-up as NaN. Filling that with 0 makes the first
    # BASELINE_MIN rows beat any multiplier, so just skip the rule until there
    # is a baseline to compare against.
    baseline = out["roll_sum_N"].rolling(BASELINE_N, min_periods=BASELINE_MIN).median()

    rules = {
        "high_z": out["amount_z"] > thresholds.z_cut,
        "rapid_swipes": out["rapid_swipes"] == 1,
        "repeat_small": out["repeat_small_charge"] == 1,
        "geo_jump": out["geo_dist_km"] > thresholds.geo_km,
        "velocity_spike": baseline.notna() & (out["roll_sum_N"] > baseline * thresholds.velocity_multiplier),
    }
    hits = pd.DataFrame({name: mask.fillna(False) for name, mask in rules.items()})

    out["reasons"] = hits.apply(lambda row: ",".join(hits.columns[row.to_numpy()]), axis=1)
    out["rule_hits"] = hits.sum(axis=1)

    cutoff = float(np.percentile(out["risk_score"], thresholds.percentile))
    min_hits = 2 if thresholds.require_two_rules else 1
    out["is_alert"] = (out["risk_score"] >= cutoff) | (out["rule_hits"] >= min_hits)
    out.attrs["risk_threshold"] = cutoff
    return out
