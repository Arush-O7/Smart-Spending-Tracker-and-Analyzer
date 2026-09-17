import numpy as np
import pandas as pd

from spending_analyzer.config import AmountBinning
from spending_analyzer.features import (
    assign_amount_bin,
    build_features,
    fit_amount_bin_edges,
    normalize_merchant,
    parse_timestamps,
)


def test_normalize_merchant():
    s = pd.Series(["fraud_Ruecker, Beer  and Collier", None, "  SHOP--A  "])
    assert normalize_merchant(s).tolist() == ["fraud_ruecker beer and collier", "", "shop a"]


def test_day_first_dates_keep_their_day():
    parsed = parse_timestamps(pd.Series(["30-06-2019 23:58", "25-12-2019 10:00"]))
    assert [(t.day, t.month) for t in parsed] == [(30, 6), (25, 12)]


def test_whole_column_uses_one_format_even_if_row_one_is_ambiguous():
    # 01-07-2019 on its own could be either, but 30-06 settles it for the column.
    parsed = parse_timestamps(pd.Series(["01-07-2019 00:00", "02-07-2019 10:00", "30-06-2019 23:58"]))
    assert [(t.day, t.month) for t in parsed] == [(1, 7), (2, 7), (30, 6)]


def test_month_first_column_stays_month_first():
    parsed = parse_timestamps(pd.Series(["07-01-2019", "12-25-2019"]))
    assert [(t.day, t.month) for t in parsed] == [(1, 7), (25, 12)]


def test_iso_dates():
    assert parse_timestamps(pd.Series(["2024-03-09 08:15:00"]))[0] == pd.Timestamp("2024-03-09 08:15")


def test_junk_dates_become_nat():
    assert parse_timestamps(pd.Series(["2024-01-01", "not a date"])).isna().tolist() == [False, True]


def test_sample_dates_land_in_the_right_months(raw_sample):
    parsed = parse_timestamps(raw_sample["trans_date_trans_time"])
    assert parsed.notna().all()
    # The sample runs from 30 June into July 2019. A month-first misread lands in January.
    assert set(parsed.dt.month.unique()) <= {6, 7}


def test_amount_bin_does_not_depend_on_neighbouring_rows():
    binning = AmountBinning([10.0, 50.0, 100.0], "test")
    alone = assign_amount_bin(pd.Series([75.0]), binning)
    together = assign_amount_bin(pd.Series([1.0, 75.0, 5000.0]), binning)
    assert alone[0] == together[1] == 2


def test_single_row_gets_a_real_bin():
    binned = assign_amount_bin(pd.Series([499.0]))
    assert binned.notna().all()
    assert binned.dtype.kind == "i"


def test_build_features_gives_the_model_everything_it_wants(transactions):
    feats, dropped, _ = build_features(transactions)
    for c in ["merchant", "log_amount", "amount_bin", "day_of_week", "hour"]:
        assert c in feats.columns
    assert feats[["log_amount", "amount_bin", "day_of_week", "hour"]].notna().all().all()
    assert dropped == 0


def test_build_features_counts_what_it_dropped():
    df = pd.DataFrame({
        "timestamp": ["2024-01-01", "garbage", "2024-01-03"],
        "merchant": ["a", "b", "c"],
        "amount": [1.0, 2.0, "not a number"],
    })
    feats, dropped, reasons = build_features(df)
    assert len(feats) == 1
    assert dropped == 2
    assert reasons == {"bad_timestamp": 1, "bad_amount": 1}


def test_build_features_on_an_empty_frame():
    feats, dropped, _ = build_features(pd.DataFrame({"timestamp": [], "merchant": [], "amount": []}))
    assert feats.empty
    assert dropped == 0


def test_fit_amount_bin_edges():
    edges = fit_amount_bin_edges(pd.Series(np.arange(1, 101, dtype=float)), n_bins=5)
    assert len(edges) == 4
    assert edges == sorted(edges)
