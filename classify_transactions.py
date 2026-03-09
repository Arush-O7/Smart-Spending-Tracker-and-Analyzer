# classify_transactions.py

import pandas as pd
import numpy as np
import joblib
import argparse # For command-line arguments
import re # Keep for potential future use

# --- Load Saved Components ---
try:
    # Load BOTH models
    rf_model = joblib.load('random_forest_model.joblib')
    mlp_model = joblib.load('mlp_model.joblib') # Load the MLP model
    scaler = joblib.load('scaler.joblib')
    le = joblib.load('label_encoder.joblib')
    print("Loaded RF model, MLP model, scaler, and label encoder.")
except FileNotFoundError as e:
    print(f"--- ERROR: Could not find a required .joblib file ({e}).")
    print("Make sure 'random_forest_model.joblib', 'mlp_model.joblib', 'scaler.joblib', and 'label_encoder.joblib' are present.")
    print("Please run the Jupyter notebook first to generate these files.")
    exit() # Stop the script
except Exception as e:
    print(f"--- ERROR loading files: {e} ---")
    exit()

# --- Utility Functions (Copied from previous version) ---
# Note: Ideally, these would be in a separate utils.py file and imported
def clean_data_types_script(df):
    """Cleans essential columns in the DataFrame for the script."""
    print("Cleaning data types...")
    df_cleaned = df.copy()
    # Find essential columns using common names/heuristics if needed
    timestamp_col = next((col for col in df.columns if 'time' in col.lower() or 'date' in col.lower()), None)
    amount_col = next((col for col in df.columns if 'amt' in col.lower() or 'amount' in col.lower()), None)
    desc_col = next((col for col in df.columns if 'desc' in col.lower() or 'merchant' in col.lower()), None)

    if not timestamp_col or not amount_col or not desc_col:
         print("--- ERROR: Could not reliably find Timestamp, Amount, or Description columns. ---")
         print(f"Found columns: {df.columns.tolist()}")
         return pd.DataFrame() # Return empty DataFrame on critical error

    # Rename for consistency
    df_cleaned.rename(columns={
        timestamp_col: 'Timestamp',
        amount_col: 'Amount',
        desc_col: 'Description'
    }, inplace=True)

    # Convert Timestamp (try common formats)
    try:
         df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'], errors='coerce', dayfirst=True, infer_datetime_format=True)
         if df_cleaned['Timestamp'].isnull().all():
              print("Timestamp parsing with dayfirst failed, trying default...")
              df_cleaned['Timestamp'] = pd.to_datetime(df_cleaned['Timestamp'], errors='coerce', infer_datetime_format=True) # Use original name before rename if needed
    except Exception as e:
         print(f"Warning: Timestamp parsing failed: {e}")
         df_cleaned['Timestamp'] = pd.NaT

    # Convert Amount
    if df_cleaned['Amount'].dtype == 'object':
         df_cleaned['Amount'] = df_cleaned['Amount'].astype(str).str.replace(r'[$,]', '', regex=True)
    df_cleaned['Amount'] = pd.to_numeric(df_cleaned['Amount'], errors='coerce')

    # Ensure Description is string
    df_cleaned['Description'] = df_cleaned['Description'].astype(str).fillna('Unknown')

    # --- Handle Missing/Infinite Values ---
    print("Handling missing/infinite values...")
    initial_rows = len(df_cleaned)
    df_cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_cleaned.dropna(subset=['Timestamp', 'Amount'], inplace=True)
    rows_dropped = initial_rows - len(df_cleaned)
    if rows_dropped > 0:
         print(f"Dropped {rows_dropped} rows due to missing Timestamps or Amounts.")

    return df_cleaned

def select_and_engineer_features_script(df, expected_features):
    """Selects and engineers features consistent with training for the script."""
    print("Selecting and engineering features...")
    # Select potentially useful NUMERIC columns available in the input df
    potential_numeric_cols = [
        'lat', 'long', 'city_pop', 'unix_time', 'merch_lat', 'merch_long',
        'Unnamed: 0', 'cc_num', 'zip', 'is_fraud', 'merch_zipcode'
    ]
    base_feature_cols = [col for col in potential_numeric_cols if col in df.columns and pd.api.types.is_numeric_dtype(df[col])]

    if not base_feature_cols:
        print("--- WARNING: No base numeric features found. Using only engineered features. ---")
        X_features = pd.DataFrame(index=df.index)
    else:
        print(f"Using base numeric features: {base_feature_cols}")
        X_features = df[base_feature_cols].copy()

    # Add engineered features
    X_features['Hour'] = df['Timestamp'].dt.hour
    X_features['DayOfWeek'] = df['Timestamp'].dt.dayofweek
    X_features['AbsAmount'] = df['Amount'].abs()

    # Fill potential NaNs
    X_features.fillna(0, inplace=True)

    # Ensure final features match the order expected by the scaler/model
    missing_features = [f for f in expected_features if f not in X_features.columns]
    for f in missing_features:
         print(f"Warning: Feature '{f}' expected but not generated/found. Adding column of zeros.")
         X_features[f] = 0

    extra_features = [f for f in X_features.columns if f not in expected_features]
    if extra_features:
         print(f"Warning: Extra features generated/found: {extra_features}. They will be ignored.")
         # X_features = X_features.drop(columns=extra_features) # Alternative

    # Reorder columns
    try:
         X_final = X_features[expected_features]
    except KeyError as e:
         print(f"--- ERROR: Could not align features. Missing expected feature: {e}")
         return pd.DataFrame()

    return X_final


# --- Main Script Logic ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Classify transactions from a CSV file using RF or MLP.")
    parser.add_argument("csv_filepath", help="Path to the input CSV file.")
    # Add argument to choose model
    parser.add_argument(
        "--model_type",
        choices=['rf', 'mlp'], # Allow rf or mlp
        default='rf',          # Default to Random Forest
        help="Specify model: 'rf' (Random Forest) or 'mlp' (MLP)."
    )
    args = parser.parse_args()

    print(f"\nProcessing file: {args.csv_filepath}")
    print(f"Using model type: {args.model_type.upper()}") # Show selected model

    # Load input CSV (Keep robust loading logic)
    try:
         input_df_orig = pd.read_csv(args.csv_filepath, header='infer', skip_blank_lines=True, on_bad_lines='skip')
         first_val = str(input_df_orig.iloc[0, 0])
         if first_val.isdigit() and len(input_df_orig.columns) > 10 and 'trans_date_trans_time' not in input_df_orig.columns:
               print("First row looks like data, attempting reload with header=1...")
               input_df = pd.read_csv(args.csv_filepath, header=1, skip_blank_lines=True, on_bad_lines='skip')
         else:
               input_df = input_df_orig
    except FileNotFoundError:
        print(f"--- ERROR: Input file not found at '{args.csv_filepath}'")
        exit()
    except Exception as e:
        print(f"--- ERROR loading input CSV: {e}")
        exit()

    # Clean and Prepare Data
    df_cleaned_input = clean_data_types_script(input_df)
    if df_cleaned_input.empty:
        print("No valid data found after cleaning. Exiting.")
        exit()

    # Select/Engineer features
    expected_feature_names = scaler.get_feature_names_out()
    X_input = select_and_engineer_features_script(df_cleaned_input, expected_feature_names)
    if X_input.empty:
         print("Could not generate features. Exiting.")
         exit()

    # Scale features
    print("Scaling input features...")
    X_input_scaled = scaler.transform(X_input)

    # --- Select and Use Model based on argument ---
    if args.model_type == 'mlp':
        print("Making predictions using MLP model...")
        model_to_use = mlp_model # Use the loaded MLP model
    else: # Default to RF
        print("Making predictions using Random Forest model...")
        model_to_use = rf_model # Use the loaded RF model

    predictions_encoded = model_to_use.predict(X_input_scaled)
    # --- End Model Selection ---

    # Decode predictions
    predictions_labels = le.inverse_transform(predictions_encoded)

    # --- Display Results ---
    print("\n--- Predictions ---")
    display_cols = ['Timestamp', 'Description', 'Amount']
    display_cols_present = [col for col in display_cols if col in df_cleaned_input.columns]
    results_df = df_cleaned_input[display_cols_present].copy()
    results_df['Predicted_Category'] = predictions_labels
    print(results_df.to_string(index=False))

    print("\nClassification complete.")