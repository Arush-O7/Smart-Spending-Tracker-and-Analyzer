import numpy as np
import pandas as pd
import pytest

from spending_analyzer.config import PROJECT_ROOT

SAMPLE_CSV = PROJECT_ROOT / "data" / "for_testing.csv"


@pytest.fixture
def raw_sample():
    if not SAMPLE_CSV.exists():
        pytest.skip(f"sample data missing: {SAMPLE_CSV}")
    return pd.read_csv(SAMPLE_CSV)


@pytest.fixture
def transactions():
    """Canonical-schema rows with one obvious monthly subscription in them."""
    rng = np.random.default_rng(1234)
    rows = []

    for i in range(6):
        rows.append({
            "timestamp": pd.Timestamp("2024-01-05 09:00") + pd.Timedelta(days=30 * i),
            "merchant": "netflix",
            "amount": 499.0,
            "city": "pune",
            "state": "mh",
        })

    for i in range(120):
        rows.append({
            "timestamp": pd.Timestamp("2024-01-01 12:00") + pd.Timedelta(hours=6 * i),
            "merchant": f"shop {i % 9}",
            "amount": float(round(abs(rng.normal(180, 60)) + 5, 2)),
            "city": "pune",
            "state": "mh",
        })

    return pd.DataFrame(rows)
