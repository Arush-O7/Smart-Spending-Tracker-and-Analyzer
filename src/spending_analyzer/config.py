"""Paths and shared constants."""

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODELS_DIR = Path(os.environ.get("SPENDING_ANALYZER_MODELS_DIR", PROJECT_ROOT / "models"))
DATA_DIR = PROJECT_ROOT / "data"

MODEL_PATH = MODELS_DIR / "categorization_pipeline.joblib"
RISK_MODEL_PATH = MODELS_DIR / "risk_engine_model.joblib"
AMOUNT_BINS_PATH = MODELS_DIR / "amount_bins.json"

REQUIRED_CANON = ["timestamp", "merchant", "amount"]
OPTIONAL_CANON = ["city", "state", "category"]
CANON_ORDER = REQUIRED_CANON + OPTIONAL_CANON

# What the categorization pipeline was trained on.
TEXT_COL = "merchant"
NUM_COLS = ["log_amount", "amount_bin", "day_of_week", "hour"]

# The notebook trains with pd.qcut(amount, q=5).
AMOUNT_BIN_COUNT = 5

# Interior cut points for amount_bin, used when models/amount_bins.json is absent.
# These are the quintiles of data/for_testing.csv, which comes from the same
# Kaggle dump as the training data. Regenerate against the full file with:
#     spending-analyzer fit-bins data/credit_card_transactions.csv
FALLBACK_BIN_EDGES = [10.29, 51.86, 71.69, 103.50]


class AmountBinning:
    """Fixed cut points used to discretise amounts."""

    def __init__(self, edges, source):
        self.edges = list(edges)
        self.source = source

    @property
    def n_bins(self):
        return len(self.edges) + 1

    def __repr__(self):
        return f"AmountBinning(edges={self.edges}, source={self.source!r})"


def load_amount_binning(path=None):
    """Read the persisted edges, falling back to the built-in ones."""
    path = Path(path or AMOUNT_BINS_PATH)
    try:
        edges = [float(e) for e in json.loads(path.read_text(encoding="utf-8"))["edges"]]
    except (OSError, ValueError, KeyError, TypeError):
        return AmountBinning(FALLBACK_BIN_EDGES, "built-in")
    if not edges or edges != sorted(edges):
        return AmountBinning(FALLBACK_BIN_EDGES, "built-in")
    return AmountBinning(edges, str(path))


def save_amount_binning(edges, path=None):
    path = Path(path or AMOUNT_BINS_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"edges": [float(e) for e in edges], "n_bins": len(edges) + 1}
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
