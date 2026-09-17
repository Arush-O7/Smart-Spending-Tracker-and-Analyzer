import numpy as np
import pandas as pd
import pytest

from spending_analyzer.models import RiskBundle
from spending_analyzer.risk import (
    RiskThresholds,
    apply_rules,
    build_risk_features,
    haversine_km,
    score_risk,
)

FEATURE_COLS = ["hour", "dayofweek", "is_weekend", "amount", "amount_z", "roll_cnt_N",
                "roll_sum_N", "roll_max_N", "prev_txn_delta_min", "merchant_conc_ewm",
                "rapid_swipes", "repeat_small_charge", "geo_dist_km"]


class FakeDetector:
    """Stands in for the Isolation Forest so these tests need no artifact."""

    def __init__(self, values):
        self.values = np.asarray(values, dtype=float)

    def decision_function(self, X):
        assert isinstance(X, pd.DataFrame), "the estimator should get named features"
        return self.values[: len(X)]


def canonical(n=60):
    return pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=n, freq="6h"),
        "merchant": [f"shop {i % 5}" for i in range(n)],
        "amount": np.linspace(10, 300, n),
    })


def test_accepts_our_own_column_names():
    feats = build_risk_features(canonical())
    for c in FEATURE_COLS:
        assert c in feats.columns


def test_accepts_the_raw_kaggle_columns(raw_sample):
    feats = build_risk_features(raw_sample)
    assert len(feats) > 0
    for c in FEATURE_COLS:
        assert c in feats.columns
    assert feats["geo_dist_km"].gt(0).any()


def test_rejects_a_file_with_no_amount():
    with pytest.raises(ValueError, match="amount"):
        build_risk_features(pd.DataFrame({"timestamp": ["2024-01-01"], "merchant": ["a"]}))


def test_haversine():
    # Mumbai to Delhi is roughly 1150 km.
    assert 1100 < haversine_km(19.076, 72.877, 28.704, 77.102) < 1200


def test_score_spans_zero_to_one():
    feats = build_risk_features(canonical())
    values = np.linspace(-0.4, 0.6, len(feats))
    scores = score_risk(feats, RiskBundle(FakeDetector(values), FEATURE_COLS))
    assert scores.min() == pytest.approx(0.0)
    assert scores.max() == pytest.approx(1.0)


def test_more_anomalous_scores_higher():
    feats = build_risk_features(canonical(10))
    values = np.array([0.5, -0.5] + [0.0] * 8)
    scores = score_risk(feats, RiskBundle(FakeDetector(values), FEATURE_COLS))
    assert scores[1] > scores[0]


def test_flat_decision_values_do_not_blow_up():
    feats = build_risk_features(canonical(10))
    scores = score_risk(feats, RiskBundle(FakeDetector(np.zeros(10)), FEATURE_COLS))
    assert (scores == 0).all()


def test_missing_feature_columns_are_filled_in():
    feats = build_risk_features(canonical(10)).drop(columns=["geo_dist_km"])
    scores = score_risk(feats, RiskBundle(FakeDetector(np.linspace(0, 1, 10)), FEATURE_COLS))
    assert len(scores) == 10


def test_velocity_rule_is_quiet_until_there_is_a_baseline():
    feats = build_risk_features(canonical(40))
    feats["risk_score"] = 0.0
    scored = apply_rules(feats, RiskThresholds(percentile=99.0))
    assert not scored["reasons"].str.contains("velocity_spike").any()


def test_rapid_micro_charges_trip_two_rules():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01 00:00", periods=12, freq="1min"),
        "merchant": ["skimmer"] * 12,
        "amount": [1.5] * 12,
    })
    feats = build_risk_features(df)
    feats["risk_score"] = 0.0
    scored = apply_rules(feats, RiskThresholds(percentile=99.0))
    assert scored["reasons"].str.contains("rapid_swipes").any()
    assert scored["reasons"].str.contains("repeat_small").any()
    assert scored["is_alert"].any()


def test_two_rules_is_stricter_than_one():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=30, freq="2h"),
        "merchant": ["shop"] * 30,
        "amount": [50.0] * 29 + [9000.0],
    })
    feats = build_risk_features(df)
    feats["risk_score"] = np.zeros(len(feats))
    strict = apply_rules(feats, RiskThresholds(percentile=100.0, require_two_rules=True))
    loose = apply_rules(feats, RiskThresholds(percentile=100.0, require_two_rules=False))
    assert strict["is_alert"].sum() <= loose["is_alert"].sum()
