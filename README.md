# 🔍 Smart Spending Tracker & Analyzer

An advanced financial analytics system that combines **Supervised Classification** with **Unsupervised Anomaly Detection** to categorize transactions and identify high-risk spending behavior.

---

## 🚀 Engineering Highlights

- **Hybrid Intelligence Model**: Integrates a Random Forest classifier for high-accuracy categorization and an **Isolation Forest** algorithm for multi-dimensional anomaly detection without labeled fraud data.
- **Big Data Optimization**: Trained on **real-world transactions**, ensuring the model handles diverse merchant naming conventions and high-volume data streams.
- **Advanced Feature Engineering**: Derived temporal features (Time-of-day, Day-of-week) and behavioral metrics (Z-score outliers, Velocity Spikes, and Geo-Jump violations) to improve prediction confidence.
- **Production-Ready UI**: Developed a full-stack interactive dashboard using **Streamlit** and **Plotly** for real-time visualization of financial health and budget variance.

---

## 📂 System Architecture

The repository is organized following professional modular software standards:

```text
.
├── models/             # Serialized ML Artifacts (.joblib)
│   ├── categorization_pipeline.joblib   # 300k record Random Forest model
│   └── risk_engine_model.joblib        # Isolation Forest anomaly detector
├── data/               # Testing & Schema samples
│   └── for_testing.csv # Anonymized demo transaction set
├── notebooks/          # Research & Development
│   └── transaction_categorization.ipynb # Model training & EDA
├── app.py              # Main Streamlit Dashboard Application
├── requirements.txt    # Dependency Management
└── README.md           # Documentation
