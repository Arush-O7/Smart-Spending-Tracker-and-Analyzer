# 📂 Models

This directory contains serialized model artifacts used for **inference**.

## Contents

- **`categorization_pipeline.joblib`**
  A `scikit-learn` pipeline that includes preprocessing steps and a trained classifier.

- **`risk_engine_model.joblib`**
  A dict of `{"pipe": <detector>, "feature_cols": [...]}` used by the risk tab.

- **`amount_bins.json`**
  The cut points for the `amount_bin` feature.

## Why the bin edges are stored here

`amount_bin` is one of the four numeric inputs to the classifier. If those edges get
recomputed from whatever file is being scored (e.g. `pd.qcut` on the upload), then the
same transaction ends up in a different bin depending on its neighbours, and a file with
fewer rows than bins just produces `NaN`, which the estimator refuses outright.

So the edges are pinned on disk. Regenerate them after retraining:

```bash
spending-analyzer fit-bins data/credit_card_transactions.csv --bins 5
```

The committed values are the quintiles of `data/for_testing.csv`, which comes from the
same Kaggle dump as the training data. Regenerate against the full file if you have it.

## Notes

- Ensure that package versions in **`requirements.txt`** remain aligned with the artifact's
  training environment.
- This is especially important for **scikit-learn**, as version mismatches may cause
  deserialization issues. `spending-analyzer show-config` checks both artifacts load.
