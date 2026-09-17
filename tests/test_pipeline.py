"""End to end against the real artifacts. Skipped if they aren't checked out."""

import numpy as np
import pandas as pd
import pytest

from spending_analyzer.features import build_features
from spending_analyzer.models import ModelLoadError, load_categorization_pipeline, load_risk_bundle
from spending_analyzer.risk import RiskThresholds, apply_rules, build_risk_features, score_risk
from spending_analyzer.schema import align_to_canonical


@pytest.fixture(scope="module")
def pipe():
    try:
        return load_categorization_pipeline()
    except ModelLoadError as e:
        pytest.skip(str(e))


@pytest.fixture(scope="module")
def risk_bundle():
    try:
        return load_risk_bundle()
    except ModelLoadError as e:
        pytest.skip(str(e))


def test_sample_accuracy_holds_up(pipe, raw_sample):
    feats, dropped, _ = build_features(align_to_canonical(raw_sample))
    assert dropped == 0
    acc = (pipe.predict(feats) == feats["category"]).mean()
    assert acc > 0.95, f"accuracy dropped to {acc:.3f}"


def test_a_single_transaction_can_be_categorized(pipe):
    df = pd.DataFrame({"timestamp": ["2024-01-01 10:00"], "merchant": ["netflix.com"],
                       "amount": [499.0]})
    feats, _, _ = build_features(align_to_canonical(df))
    assert len(pipe.predict(feats)) == 1


def test_prediction_does_not_move_when_other_rows_change(pipe):
    one = pd.DataFrame({"timestamp": ["2024-01-01 10:00"], "merchant": ["netflix.com"],
                        "amount": [499.0]})
    many = pd.concat([one, pd.DataFrame({
        "timestamp": pd.date_range("2024-01-02", periods=50, freq="D").astype(str),
        "merchant": ["random shop"] * 50,
        "amount": np.linspace(1, 100000, 50),
    })], ignore_index=True)

    alone, _, _ = build_features(align_to_canonical(one))
    together, _, _ = build_features(align_to_canonical(many))
    assert pipe.predict(alone)[0] == pipe.predict(together)[0]


def test_real_risk_scores_stay_in_range(risk_bundle, raw_sample):
    scores = score_risk(build_risk_features(raw_sample), risk_bundle)
    assert scores.min() >= 0.0
    assert scores.max() <= 1.0


def test_alert_rate_is_sane(risk_bundle, raw_sample):
    feats = build_risk_features(raw_sample)
    feats["risk_score"] = score_risk(feats, risk_bundle)
    scored = apply_rules(feats, RiskThresholds(percentile=99.0, require_two_rules=True))
    rate = scored["is_alert"].mean()
    assert 0 < rate < 0.25, f"alert rate of {rate:.1%} looks mis-calibrated"
