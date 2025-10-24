"""
autofill.py
------------
Handles automatic filling of missing or incomplete LCA input data.

Uses:
- Trained iterative imputation model (XGBoost + MICE)
- Stored label encoders for categorical columns
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

import warnings
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
warnings.filterwarnings("ignore", message=".*XGBoost.*")  # silences the XGBoost version warning

# ---------------------------------------------
# Define paths
# ---------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")
DATA_DIR = os.path.join(BASE_DIR, "data")

# Load saved model & encoders
IMPUTER_PATH = os.path.join(MODEL_DIR, "xgb_imputer.pkl")  # <- only XGB model
ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
DATASET_PATH = os.path.join(DATA_DIR, "lca_dataset.csv")

# Load imputer and encoders
imputer = joblib.load(IMPUTER_PATH) if os.path.exists(IMPUTER_PATH) else None
label_encoders = joblib.load(ENCODERS_PATH) if os.path.exists(ENCODERS_PATH) else {}

# Load training schema if available
if os.path.exists(DATASET_PATH):
    df_training = pd.read_csv(DATASET_PATH)
else:
    df_training = pd.DataFrame()

# Identify categorical & numeric columns
categorical_cols = list(label_encoders.keys()) if label_encoders else df_training.select_dtypes(include=['object']).columns.tolist()
numeric_cols = df_training.select_dtypes(include=[np.number]).columns.tolist() if not df_training.empty else []


# ---------------------------------------------
# Core Autofill Function (model-only)
# ---------------------------------------------
def autofill_missing_values(input_dict: dict) -> dict:
    """
    Autofills missing numeric/categorical values using the trained XGB imputer.

    Args:
        input_dict (dict): Input data (may have nulls)

    Returns:
        dict: Fully filled dictionary (model-imputed only)
    """
    try:
        # Step 1 — Convert input to DataFrame
        df_user = pd.DataFrame([input_dict])

        # Step 2 — Align with training schema
        if not df_training.empty:
            df_user = df_user.reindex(columns=df_training.columns, fill_value=np.nan)

        # Step 3 — Encode categorical columns
        for col in categorical_cols:
            if col in df_user.columns and col in label_encoders:
                le = label_encoders[col]
                df_user[col] = df_user[col].map(
                    lambda x: le.transform([x])[0] if pd.notna(x) and x in le.classes_ else -1
                )

        # Step 4 — Impute missing values
        if imputer is not None:
            imputed_array = imputer.transform(df_user)
            df_imputed = pd.DataFrame(imputed_array, columns=df_user.columns)
        else:
            raise RuntimeError("❌ No imputer found. Please train and save 'xgb_imputer.pkl'.")

        # Step 5 — Decode categorical columns back
        for col in categorical_cols:
            if col in label_encoders:
                le = label_encoders[col]
                df_imputed[col] = df_imputed[col].round().astype(int)
                df_imputed[col] = df_imputed[col].map(
                    lambda x: le.classes_[x] if x < len(le.classes_) else "Unknown"
                )

        # Step 6 — Return as dictionary
        return df_imputed.iloc[0].to_dict()

    except Exception as e:
        raise RuntimeError(f"Error during model-based autofill: {str(e)}")


# ---------------------------------------------
# Test block (remove in production)
# ---------------------------------------------
if __name__ == "__main__":
    sample_input = {
        'Process Stage': 'Manufacturing',
        'Technology': 'Emerging',
        'Time Period': '2020-2025',
        'Location': 'Asia',
        'Functional Unit': '1 kg Aluminium Sheet',
        'Raw Material Type': 'Aluminium Scrap',
        'Raw Material Quantity (kg or unit)': 100.0,
        'Energy Input Type': 'Electricity',
        'Energy Input Quantity (MJ)': 250.0,
        'Processing Method': 'Advanced',
        'Transport Mode': None,
        'Transport Distance (km)': 300.0,
        'Fuel Type': 'Diesel',
        'Metal Quality Grade': 'High',
        'Material Scarcity Level': 'Medium',
        'Material Cost (USD)': 500.0,
        'Processing Cost (USD)': 200.0,
        'End-of-Life Treatment': 'Recycling',
        'Emissions to Air CO2 (kg)': None,
        'Emissions to Air SOx (kg)': None,
        'Emissions to Air NOx (kg)': None,
        'Emissions to Air Particulate Matter (kg)': None,
        'Greenhouse Gas Emissions (kg CO2-eq)': None,
        'Scope 1 Emissions (kg CO2-eq)': None,
        'Scope 2 Emissions (kg CO2-eq)': None,
        'Scope 3 Emissions (kg CO2-eq)': None,
        'Environmental Impact Score': None,
        'Metal Recyclability Factor': None
    }

    result = autofill_missing_values(sample_input)
    print("\nAutofilled Result:")
    print(json.dumps(result, indent=2))
