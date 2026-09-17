# 📂 Data

This directory contains datasets for **training** and **testing**.

## Supported Schema (Flexible)

- **Required**: `timestamp`, `merchant`, `amount`
- **Optional**: `city`, `state`, `category`
- Common header variants are auto-detected (e.g. `amt` / `amnt` / `amount`; `trans_date_trans_time` / `datetime` / `timestamp`). The full alias list is in the README.

Risk scoring can also use `cc_num` (to keep the rolling windows per card) and
`lat` / `long` / `merch_lat` / `merch_long` (for the geo jump rule).

## Files

- **`for_testing.csv`**
  Small sample used for verifying the **Streamlit app** and by the test suite.
  Dates in it are day-first (`DD-MM-YYYY`).

- **`credit_card_transactions.csv`** _(not included in repo)_
  Full dataset used for **training** the model.
  - 📥 Download from Kaggle: [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
  - After downloading, place the file in this `data/` folder. It's gitignored.

## Example Usage

```bash
spending-analyzer predict data/for_testing.csv -n 10
```

```python
import pandas as pd

# Load training data (after manual download)
train_df = pd.read_csv("data/credit_card_transactions.csv")

# Load small test sample
test_df = pd.read_csv("data/for_testing.csv")
```
