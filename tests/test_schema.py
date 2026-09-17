import pandas as pd
import pytest

from spending_analyzer.schema import SchemaError, align_to_canonical, guess_schema_columns


def test_detects_raw_kaggle_headers(raw_sample):
    mapping, missing = guess_schema_columns(raw_sample)
    assert missing == []
    assert mapping["timestamp"] == "trans_date_trans_time"
    assert mapping["amount"] == "amt"
    assert mapping["merchant"] == "merchant"


def test_aligns_raw_headers(raw_sample):
    aligned = align_to_canonical(raw_sample)
    assert list(aligned.columns) == ["timestamp", "merchant", "amount", "city", "state", "category"]
    assert len(aligned) == len(raw_sample)


def test_canonical_input_passes_through():
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "merchant": ["a"], "amount": [1.0]})
    assert list(align_to_canonical(df).columns) == ["timestamp", "merchant", "amount"]


@pytest.mark.parametrize("headers", [
    ("datetime", "description", "value"),
    ("Transaction Date", "Narration", "Debit"),
    ("TXN_TIME", "Payee", "Txn Amount"),
])
def test_common_bank_export_aliases(headers):
    ts, merchant, amount = headers
    df = pd.DataFrame({ts: ["2024-01-01"], merchant: ["shop"], amount: [10.0]})
    assert set(align_to_canonical(df).columns) == {"timestamp", "merchant", "amount"}


def test_missing_column_error_says_what_it_saw():
    df = pd.DataFrame({"when": ["2024-01-01"], "who": ["shop"]})
    with pytest.raises(SchemaError) as e:
        align_to_canonical(df)
    assert "amount" in str(e.value)
    assert "Columns found in the file" in str(e.value)


def test_one_column_is_not_claimed_twice():
    # "category" is canonical and "class" is an alias for it.
    df = pd.DataFrame({"timestamp": ["2024-01-01"], "merchant": ["a"], "amount": [1.0],
                       "category": ["food"], "class": ["x"]})
    mapping, _ = guess_schema_columns(df)
    claimed = [v for v in mapping.values() if v]
    assert len(claimed) == len(set(claimed))
    assert mapping["category"] == "category"
