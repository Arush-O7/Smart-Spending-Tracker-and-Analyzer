import time

import numpy as np
import pandas as pd

from spending_analyzer.anomaly import detect_multi_dimensional_anomalies
from spending_analyzer.insights import make_facts


def noise(n, seed=7):
    rng = np.random.default_rng(seed)
    return pd.DataFrame({
        "timestamp": pd.Timestamp("2024-01-01")
                     + pd.to_timedelta(rng.integers(0, 365 * 24 * 60, n), unit="m"),
        "merchant": [f"m{i}" for i in rng.integers(0, 40, n)],
        "amount": np.abs(rng.normal(200, 60, n)) + 5,
        "pred_category": [f"c{i}" for i in rng.integers(0, 5, n)],
    })


def test_too_few_rows():
    res = detect_multi_dimensional_anomalies(make_facts(noise(5)))
    assert res.anomalies.empty
    assert res.total_anomalies == 0


def test_total_is_the_real_total_not_the_table_length():
    res = detect_multi_dimensional_anomalies(make_facts(noise(2000)),
                                             contamination=0.1, max_results=50)
    assert res.total_anomalies > 50
    assert len(res.anomalies) == 50
    assert res.is_truncated


def test_catches_an_obvious_outlier():
    df = noise(300)
    df.loc[0, ["amount", "merchant"]] = [500000.0, "once only merchant"]
    df.loc[0, "timestamp"] = pd.Timestamp("2024-06-01 03:30")
    res = detect_multi_dimensional_anomalies(make_facts(df), contamination=0.05, max_results=10)
    assert 500000.0 in res.anomalies["amount"].tolist()


def test_every_row_gets_a_readable_reason():
    res = detect_multi_dimensional_anomalies(make_facts(noise(400)), contamination=0.05)
    seen = set()
    for value in res.anomalies["anomaly_reason"]:
        assert value
        assert not value.endswith(",")
        seen.update(value.split(","))
    assert seen <= {"rare_merchant", "high_amount", "high_frequency_day",
                    "unusual_hour", "pattern_based"}


def test_same_input_same_output():
    facts = make_facts(noise(500))
    a = detect_multi_dimensional_anomalies(facts, contamination=0.05)
    b = detect_multi_dimensional_anomalies(facts, contamination=0.05)
    pd.testing.assert_frame_equal(a.anomalies, b.anomalies)


def test_stays_fast_on_a_big_file():
    facts = make_facts(noise(20_000))
    start = time.perf_counter()
    res = detect_multi_dimensional_anomalies(facts, contamination=0.05)
    elapsed = time.perf_counter() - start
    assert res.total_rows == len(facts)
    assert elapsed < 5.0, f"took {elapsed:.1f}s on 20k rows"
