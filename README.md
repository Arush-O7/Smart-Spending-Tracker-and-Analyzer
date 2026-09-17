# Smart Spending Tracker & Analyzer

![CI](https://github.com/Arush-O7/Smart-Spending-Tracker-and-Analyzer/actions/workflows/ci.yml/badge.svg)
![Python](https://img.shields.io/badge/python-3.10%20|%203.11%20|%203.12-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## 📝 Project Overview
This project is an intelligent financial tool designed to help users understand their spending habits through automated analysis. It combines **Machine Learning** for transaction classification and **Anomaly Detection** to identify unusual financial activity.

The system is built to handle real-world data volumes, with core models trained to ensure high accuracy across various merchant types and spending behaviors.

---

## 🚀 Getting Started

```bash
git clone https://github.com/Arush-O7/Smart-Spending-Tracker-and-Analyzer.git
cd Smart-Spending-Tracker-and-Analyzer

python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
pip install -e .

streamlit run app.py
```

The dashboard comes up on <http://localhost:8501>. Upload a CSV in the sidebar, or just point it at `data/for_testing.csv` to see what it does.

There is a CLI too, for when you don't need the dashboard:

```bash
spending-analyzer predict data/for_testing.csv
spending-analyzer predict transactions.csv -o predictions.csv
spending-analyzer risk transactions.csv --percentile 99.5
spending-analyzer show-config          # checks the models actually load
```

---

## 📂 Repository Structure

* **`app.py`**: The **Streamlit** dashboard. It only handles layout and charts now; all the analysis lives in the package below.
* **`src/spending_analyzer/`**: The actual logic, split into `schema` (header detection), `features` (cleaning and feature engineering), `insights` (recurring charges, patterns, projections), `anomaly`, `risk`, `models` (artifact loading) and `cli`.
* **`models/`**: All serialized artifacts (`.joblib`), plus `amount_bins.json`.
* **`notebooks/`**: The Jupyter notebook (`transaction_categorization.ipynb`) used for data cleaning, EDA and model training.
* **`tests/`**: Unit tests for every analysis function, plus smoke tests that actually run the Streamlit script.
* **`data/`**: Sample data (`for_testing.csv`), a curated sample from the same Kaggle source used for model training.

---

## 📥 Input Format

Only three fields are needed. Headers are auto-detected, so most bank exports and the raw Kaggle dump work without editing anything.

| Column | Required | Some accepted aliases |
|---|---|---|
| `timestamp` | yes | `trans_date_trans_time`, `datetime`, `date`, `txn_time` |
| `merchant` | yes | `merchant`, `description`, `narration`, `payee` |
| `amount` | yes | `amt`, `value`, `debit`, `transaction_amount` |
| `city` | no | `merchant_city`, `billing_city` |
| `state` | no | `merchant_state`, `region`, `province` |
| `category` | no | `label`, `class` (only used to report agreement) |

Both `30-06-2019` and `2019-06-30` work. The format is worked out once for the whole column rather than row by row, so a day-first export doesn't get half-read as month-first.

The geo rules in risk detection additionally need `lat`, `long`, `merch_lat` and `merch_long`.

---

## ✨ Key Features

### 1. AI-Driven Transaction Categorization
Unlike traditional keyword-based filters, this project uses a **TF-IDF + Logistic Regression** pipeline. It analyzes the relationship between merchant descriptions and transaction amounts to predict categories (e.g. Food, Travel, Shopping) with high confidence. Macro-F1 is **0.991** on a held-out split of 300k rows (`outputs/metrics_v2_300k.json`).

One of the model's inputs is `amount_bin`, a bucketed version of the amount. Those bucket edges are kept in `models/amount_bins.json` instead of being recomputed from whatever file you upload, otherwise the same ₹499 charge lands in different buckets depending on what else is in the file. If you retrain, regenerate them:

```bash
spending-analyzer fit-bins data/credit_card_transactions.csv --bins 5
```

### 2. Risk & Anomaly Detection
The system integrates an **Isolation Forest**, an unsupervised learning algorithm. This engine identifies "outliers"—spending events that deviate significantly from a user's established historical patterns.

Every flagged transaction comes with a reason (`rare_merchant`, `high_amount`, `high_frequency_day`, `unusual_hour`), and the risk tab layers five tunable rules on top of the model score: `high_z`, `rapid_swipes`, `repeat_small`, `geo_jump` and `velocity_spike`. Each run can be exported with a JSON manifest of the exact thresholds used, so an alert list can be reproduced later.

### 3. Interactive Budgeting Dashboard
A dedicated "Budget vs Actual" tab allows users to set monthly limits for specific categories. The dashboard provides real-time visual feedback on spending variance and remaining funds.

### 4. Patterns & Projections
Recurring charge detection (3+ charges to the same merchant, 26-33 days apart, steady amount), weekday vs weekend splits, time-of-day breakdowns, merchant diversity, and a simple projection of next month's spend. Everything is anchored to the newest date in your file, not today's date, so old exports still make sense.

---

## 🧪 Development

```bash
pip install -r requirements-dev.txt
pip install -e .

make test     # pytest with coverage
make lint     # ruff
make app      # streamlit run app.py
```

The suite covers schema detection, feature engineering and every analysis function, and also runs `app.py` itself through `streamlit.testing.AppTest` so a broken tab shows up in CI instead of in the browser.

> **Note on scikit-learn:** it's pinned to 1.6.1 in `requirements.txt`. The `.joblib` files were trained on that version and don't reliably deserialize on others, so install from the requirements file rather than upgrading it by hand.

---

## 🛡️ Privacy

Uploaded files are processed by the server running the app, kept in memory for the session and dropped when it ends. Nothing is written to disk or sent anywhere else, and the models are pre-trained so they never learn from your data. Do keep in mind that if you deploy this somewhere hosted, that host is still a third party, so for real statements run it locally.

---

## 📄 License

MIT. See [LICENSE](LICENSE).
