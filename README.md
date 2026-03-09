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
├── models/             
│   ├── categorization_pipeline.joblib   
│   └── risk_engine_model.joblib        
├── data/               
│   └── for_testing.csv 
├── notebooks/          
│   └── transaction_categorization.ipynb 
├── app.py              
├── requirements.txt    
└── README.md           
