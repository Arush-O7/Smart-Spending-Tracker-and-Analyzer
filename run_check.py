# run_check.py
# Print predictions to terminal only; no files are saved.

import pandas as pd
import numpy as np
import joblib

# -------- CONFIG: update the CSV path if needed --------
MODEL_PATH = "models/categorization_pipeline.joblib"
# If your CSV has raw headers like: trans_date_trans_time, merchant, category, amt, city, state
INPUT_CSV = "data/for_testing.csv"   # <-- set to your actual CSV path
HAS_RAW_SCHEMA = True  # True for raw headers; False if already aligned (timestamp, merchant, amount, city, state)

def align_schema(df: pd.DataFrame, has_raw: bool) -> pd.DataFrame:
    if has_raw:
        keep_map = {
            "trans_date_trans_time": "timestamp",
            "merchant": "merchant",
            "category": "category",  # optional reference; not used for predict
            "amt": "amount",
            "city": "city",
            "state": "state",
        }
        missing = [c for c in keep_map if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required raw columns: {missing}")
        df = df[list(keep_map.keys())].rename(columns=keep_map)
    else:
        req = ["timestamp","merchant","amount"]
        missing = [c for c in req if c not in df.columns]
        if missing:
            raise ValueError(f"Missing aligned columns: {missing} (need at least {req})")
        # keep optional columns if present
        opt = [c for c in ["city","state","category"] if c in df.columns]
        df = df[req + opt]
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # parse and clean
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce")
    df["merchant"] = (
        df["merchant"].fillna("")
        .str.lower()
        .str.replace(r"[^\w\s]", " ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )
    # drop rows missing essentials
    df = df.dropna(subset=["timestamp","amount"]).reset_index(drop=True)
    # features
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["hour"] = df["timestamp"].dt.hour
    df["log_amount"] = np.log1p(df["amount"])
    # quick amount_bin for CLI check
    try:
        df["amount_bin"] = pd.qcut(df["amount"], q=4, labels=False, duplicates="drop")
    except Exception:
        df["amount_bin"] = pd.cut(
            df["amount"],
            bins=[-np.inf, 10, 50, 200, np.inf],
            labels=False,
            include_lowest=True
        )
    return df

def main():
    # load model
    pipe = joblib.load(MODEL_PATH)

    # read CSV
    raw = pd.read_csv(INPUT_CSV)

    # align schema and build features
    df = align_schema(raw, HAS_RAW_SCHEMA)
    feat = build_features(df)

    # predict
    preds = pipe.predict(feat)

    # print compact view to terminal
    out = df.copy()
    out["pred_category"] = preds
    cols = [c for c in ["timestamp","merchant","amount","city","state","pred_category"] if c in out.columns]
    print(out[cols].head(50).to_string(index=False))

if __name__ == "__main__":
    main()
