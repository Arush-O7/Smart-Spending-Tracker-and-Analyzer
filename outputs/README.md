# 📂 Outputs

This directory contains **lightweight, shareable artifacts** created during development or
evaluation. These files help with quick inspection, debugging, and reproducibility.

## Files

- **`metrics_v2_300k.json`**
  Validation metrics for the shipped model: macro-F1 0.991, weighted-F1 0.992.

- **`confusion_matrix_v2_300k.png`**
  Diagnostic plot from that same run.

- **`confusion_matrix.png`**
  From an earlier run, kept around for comparison.

- **`predictions(for_testing).csv`**
  Predictions over `data/for_testing.csv`. Regenerate with:

  ```bash
  spending-analyzer predict data/for_testing.csv -o "outputs/predictions(for_testing).csv"
  ```
