"""Smart Spending Tracker and Analyzer.

The Streamlit app is just a view over this package; everything it calculates
lives in here so it can be tested without a browser.
"""

__version__ = "1.0.0"

from .anomaly import AnomalyResult, detect_multi_dimensional_anomalies
from .config import AmountBinning, load_amount_binning, save_amount_binning
from .features import (
    assign_amount_bin,
    build_features,
    clean_transactions,
    fit_amount_bin_edges,
    normalize_merchant,
    parse_timestamps,
)
from .insights import (
    analyze_spending_patterns,
    anomaly_high_spend,
    generate_predictive_insights,
    make_facts,
    recurring_detection,
)
from .models import ModelLoadError, RiskBundle, load_categorization_pipeline, load_risk_bundle
from .risk import RiskThresholds, apply_rules, build_risk_features, score_risk
from .schema import SchemaError, align_to_canonical, guess_schema_columns
