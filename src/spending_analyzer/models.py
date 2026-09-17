"""Loading the joblib artifacts."""

from pathlib import Path

import joblib

from .config import MODEL_PATH, RISK_MODEL_PATH


class ModelLoadError(RuntimeError):
    pass


def _load(path, what):
    path = Path(path)
    if not path.exists():
        raise ModelLoadError(f"{what} not found at '{path}'.")
    try:
        return joblib.load(path)
    except Exception as exc:
        raise ModelLoadError(
            f"Could not load {what} from '{path}': {exc}. Usually this means the "
            "installed scikit-learn does not match the training environment, so "
            "install the pinned requirements.txt."
        ) from exc


def load_categorization_pipeline(path=None):
    return _load(path or MODEL_PATH, "categorization pipeline")


class RiskBundle:
    """The risk artifact: a scorer plus the feature order it expects."""

    def __init__(self, pipeline, feature_cols):
        self.pipeline = pipeline
        self.feature_cols = list(feature_cols)


def load_risk_bundle(path=None):
    bundle = _load(path or RISK_MODEL_PATH, "risk engine model")
    if not isinstance(bundle, dict) or "pipe" not in bundle or "feature_cols" not in bundle:
        raise ModelLoadError("Risk artifact should be a dict with 'pipe' and 'feature_cols'.")
    return RiskBundle(bundle["pipe"], bundle["feature_cols"])
