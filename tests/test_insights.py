import pandas as pd

from spending_analyzer.insights import (
    analyze_spending_patterns,
    anomaly_high_spend,
    generate_predictive_insights,
    make_facts,
    recurring_detection,
)


def test_finds_a_monthly_subscription(transactions):
    rec = recurring_detection(make_facts(transactions))
    assert "netflix" in rec["merchant"].tolist()

    row = rec.loc[rec["merchant"] == "netflix"].iloc[0]
    assert row["count"] == 6
    assert row["median_interval_days"] == 30.0
    assert row["amount_cv"] == 0.0
    assert row["next_estimated"] == row["last_seen"] + pd.Timedelta(days=30)


def test_ignores_irregular_merchants(transactions):
    rec = recurring_detection(make_facts(transactions))
    assert not rec["merchant"].str.startswith("shop").any()


def test_works_when_there_is_only_one_merchant():
    df = pd.DataFrame({
        "timestamp": [pd.Timestamp("2024-01-01") + pd.Timedelta(days=30 * i) for i in range(4)],
        "merchant": ["gym"] * 4,
        "amount": [1000.0] * 4,
    })
    assert recurring_detection(make_facts(df))["merchant"].tolist() == ["gym"]


def test_empty_input_still_has_the_right_columns():
    rec = recurring_detection(pd.DataFrame())
    assert rec.empty
    assert "median_interval_days" in rec.columns


def test_high_spend_ranks_within_its_own_category():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=6, freq="D"),
        "merchant": list("abcdef"),
        "amount": [100.0, 110.0, 105.0, 95.0, 100.0, 900.0],
        "pred_category": ["food"] * 6,
    })
    top = anomaly_high_spend(make_facts(df), top_n=1)
    assert top.iloc[0]["amount"] == 900.0
    assert top.iloc[0]["z"] > 1


def test_patterns_leave_the_input_frame_alone(transactions):
    facts = make_facts(transactions)
    before = list(facts.columns)
    analyze_spending_patterns(facts)
    assert list(facts.columns) == before


def test_patterns_keys(transactions):
    p = analyze_spending_patterns(make_facts(transactions))
    assert p["merchant_diversity_score"] > 0
    assert set(p["time_distribution"]) == {"sum", "count"}
    assert p["spending_trend"] in {"increasing", "decreasing", "stable"}


def test_revisit_is_measured_against_the_file_not_today():
    # A 2019 export. If "now" were the wall clock every merchant would look overdue.
    rows = [{
        "timestamp": pd.Timestamp("2019-01-01") + pd.Timedelta(days=7 * i),
        "merchant": "weekly cafe",
        "amount": 250.0,
        "pred_category": "food",
    } for i in range(40)]
    facts = make_facts(pd.DataFrame(rows))
    insights = generate_predictive_insights(facts)

    assert insights["as_of"] == facts["timestamp"].max()
    assert insights["merchants_due_for_revisit"] == []


def test_projections_need_enough_rows():
    df = pd.DataFrame({
        "timestamp": pd.date_range("2024-01-01", periods=5, freq="D"),
        "merchant": ["a"] * 5,
        "amount": [10.0] * 5,
    })
    assert generate_predictive_insights(make_facts(df)) == {}
