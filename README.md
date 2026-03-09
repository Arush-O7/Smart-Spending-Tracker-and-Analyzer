# Smart Spending Tracker & Analyzer

## 📝 Project Overview
This project is an intelligent financial tool designed to help users understand their spending habits through automated analysis. It combines **Machine Learning** for transaction classification and **Anomaly Detection** to identify unusual financial activity.

The system is built to handle real-world data volumes, with core models trained to ensure high accuracy across various merchant types and spending behaviors.

---

## 📂 Repository Structure
The project follows a modular structure to separate research, models, and application logic:

* **`app.py`**: The primary application file. It hosts the **Streamlit** dashboard where users upload CSV files for real-time analysis.
* **`models/`**: A centralized folder containing all serialized machine learning artifacts (`.joblib` files), including the categorization pipeline, risk engine, and data scalers.
* **`notebooks/`**: Contains the Jupyter notebook (`transaction_categorization.ipynb`) used for data cleaning, exploratory data analysis, and model training.
* **`data/`**: Includes sample data (`for_testing.csv`) for demonstration purposes. This test data is a curated sample from the same Kaggle source used for model training.

---

## ✨ Key Features

### 1. AI-Driven Transaction Categorization
Unlike traditional keyword-based filters, this project uses a **Random Forest** pipeline. It analyzes the relationship between merchant descriptions and transaction amounts to predict categories (e.g., Food, Travel, Shopping) with high confidence.

### 2. Risk & Anomaly Detection
The system integrates an **Isolation Forest** model, an unsupervised learning algorithm. This engine identifies "outliers"—spending events that deviate significantly from a user's established historical patterns.

### 3. Interactive Budgeting Dashboard
A dedicated "Budget vs Actual" tab allows users to set monthly limits for specific categories. The dashboard provides real-time visual feedback on spending variance and remaining funds.
