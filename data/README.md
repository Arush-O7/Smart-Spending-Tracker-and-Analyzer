# 📂 Data

This directory contains datasets for **training** and **testing**.

## Supported Schema (Flexible)

- **Required**: `timestamp`, `merchant`, `amount`
- **Optional**: `city`, `state`, `category`
- Common header variants are auto-detected (e.g., `amt` / `amnt` / `amount`; `trans_date_trans_time` / `datetime` / `timestamp`).

## Files

- **`for_testing.csv`**  
  Small sample used for verifying the **Streamlit app**.

- **`credit_card_transactions.csv`** _(not included in repo)_  
  Full dataset used for **training** the model.
  - 📥 Download from Kaggle: [Credit Card Transactions Dataset](https://www.kaggle.com/datasets/priyamchoksi/credit-card-transactions-dataset)
  - After downloading, place the file in this `data/` folder.

## Example Usage

```python
import pandas as pd

# Load training data (after manual download)
train_df = pd.read_csv("data/credit_card_transactions.csv")

# Load small test sample
test_df = pd.read_csv("data/for_testing.csv")
```
