# 📝 Transaction Categorization — Notebook

This notebook prepares and validates a **clean dataset** for training and evaluating a transaction categorization model.  
It handles selection, renaming, cleaning, and validation of core fields from the raw transactions file.

---

## 🎯 Objectives

- Load the raw transactions CSV and select only the fields needed for modeling.
- Standardize column names and perform basic cleaning.
- Validate the resulting dataset with quick sanity checks.
- Produce a clean, analysis-ready dataframe for downstream modeling.

---

## 📥 Data Inputs

- **Source file:** `credit_card_transactions.csv`
- **Expected fields:**
  - `trans_date_trans_time` → transaction timestamp
  - `merchant` → merchant description/text
  - `category` → ground-truth label
  - `amt` → transaction amount
  - `city` → optional, retained
  - `state` → optional, retained

⚠️ Any additional fields in the raw CSV are ignored by this baseline workflow.

---

## 🔄 Column Selection & Renaming

- `trans_date_trans_time` → `timestamp`
- `merchant` → `merchant`
- `category` → `category`
- `amt` → `amount`
- `city` → `city`
- `state` → `state`

---

## 🧹 Cleaning Steps

- Parse **timestamp** to datetime.
- Convert **amount** to numeric.
- Normalize **merchant** text: lowercase + strip punctuation/whitespace.
- Drop rows missing essentials (`timestamp`, `amount`, `category`).

---

## ✅ Sanity Checks

- Shape and dtypes of cleaned dataframe.
- Null counts across key fields.
- Preview of cleaned rows.
- Category distribution (top labels & unique count).
- Amount summary statistics.
- Duplicate row counts after selection.

---

## 📤 Outputs

- A cleaned **pandas DataFrame** in memory with columns:  
  `timestamp, merchant, category, amount, city, state`
- ⚠️ No files are written by default — export artifacts explicitly if needed.

---

## 💡 Why These Fields

- **timestamp** → derive temporal features (day-of-week, hour).
- **merchant** → primary text signal for classification (TF-IDF, hashing, etc.).
- **category** → supervised learning label.
- **amount** → numeric feature (log scaling, binning).
- **city/state** → optional geographic signals.

---

## 📝 Notes & Recommendations

- Drop stray `Unnamed` columns (from Excel/CSV exports) before selection.
- Keep environment aligned with **requirements.txt** to ensure consistent parsing/dtypes.
- Extend feature selection (ZIP, lat/long, etc.) as needed and retrain the downstream model.

---

## 🚀 Next Steps

- Use cleaned dataframe to **train or evaluate the scikit-learn pipeline** used by the Streamlit app.
- Save trained pipeline artifacts to **`models/`** for use in the app and CLI.
- Export the `amount_bin` cut points too, otherwise inference can't reproduce the binning the
  model was trained on:

  ```bash
  spending-analyzer fit-bins data/credit_card_transactions.csv --bins 5
  ```

  That writes `models/amount_bins.json`, which the app and CLI read on startup. Verify with
  `spending-analyzer show-config`, then re-run `pytest` — `tests/test_pipeline.py` checks that
  the sample accuracy hasn't regressed.
