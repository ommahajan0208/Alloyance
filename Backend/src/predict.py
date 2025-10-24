"""
predict.py
-----------
Generates sustainability metric predictions using trained XGBoost models.

Each model predicts a specific KPI:
- model_recycled_content.pkl
- model_resource_efficiency.pkl
- model_extended_product_life.pkl
- model_recovery_rate.pkl
- model_reuse_potential.pkl
"""

import os
import joblib
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
from sklearn.exceptions import InconsistentVersionWarning
warnings.filterwarnings("ignore", category=InconsistentVersionWarning)
import xgboost as xgb
xgb.set_config(verbosity=0)


# ------------------------------------------------------
# Define paths
# ------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "model")

# Map each KPI to its model filename
MODEL_PATHS = {
    "Recycled Content (%)": os.path.join(MODEL_DIR, "model_recycled_content.pkl"),
    "Resource Efficiency (%)": os.path.join(MODEL_DIR, "model_resource_efficiency.pkl"),
    "Extended Product Life (years)": os.path.join(MODEL_DIR, "model_extended_product_life.pkl"),
    "Recovery Rate (%)": os.path.join(MODEL_DIR, "model_recovery_rate.pkl"),
    "Reuse Potential (%)": os.path.join(MODEL_DIR, "model_reuse_potential.pkl"),
}

# Load available models
MODELS = {}
for name, path in MODEL_PATHS.items():
    if os.path.exists(path):
        MODELS[name] = joblib.load(path)


# ------------------------------------------------------
# Core Prediction Function
# ------------------------------------------------------
def make_prediction(autofilled_dict: dict) -> dict:
    """
    Generates all KPI predictions and merges them into a single dictionary.
    """
    try:
        if not MODELS:
            raise RuntimeError("No prediction models found in 'model/' directory.")

        df_input = pd.DataFrame([autofilled_dict])

        # ------------------------------------------
        # STEP 1 — Encode categorical columns
        # ------------------------------------------
        ENCODERS_PATH = os.path.join(MODEL_DIR, "label_encoders.pkl")
        if os.path.exists(ENCODERS_PATH):
            label_encoders = joblib.load(ENCODERS_PATH)
            for col, le in label_encoders.items():
                if col in df_input.columns:
                    df_input[col] = df_input[col].map(
                        lambda x: le.transform([x])[0] if x in le.classes_ else -1
                    )

        # ------------------------------------------
        # STEP 2 — Predict for each KPI
        # ------------------------------------------
        predictions = {}
        all_kpis = list(MODEL_PATHS.keys())

        for metric, model in MODELS.items():
            try:
                # Exclude target KPIs from features
                X_input = df_input.drop(columns=all_kpis, errors="ignore")

                y_pred = model.predict(X_input)
                predictions[metric] = float(y_pred[0])
            except Exception as e:
                predictions[metric] = None
                print(f"[WARN] Failed to predict {metric}: {e}")

        # ------------------------------------------
        # STEP 3 — Merge input + predictions
        # ------------------------------------------
        full_output = {**autofilled_dict, **predictions}
        return full_output

    except Exception as e:
        raise RuntimeError(f"Error during prediction: {str(e)}")

# ------------------------------------------------------
# Test block (for manual debugging)
# ------------------------------------------------------
if __name__ == "__main__":
    sample_input = { 
        "Process Stage": "Manufacturing", 
        "Technology": "Emerging", 
        "Time Period": "2020-2025", 
        "Location": "Asia", 
        "Functional Unit": "1 kg Aluminium Sheet", 
        "Raw Material Type": "Aluminium Scrap", 
        "Raw Material Quantity (kg or unit)": 100.0, 
        "Energy Input Type": "Electricity", 
        "Energy Input Quantity (MJ)": 250.0, 
        "Processing Method": "Advanced", 
        "Transport Mode": "Truck", 
        "Transport Distance (km)": 300.0, 
        "Fuel Type": "Diesel", 
        "Metal Quality Grade": "High", 
        "Material Scarcity Level": "Medium", 
        "Material Cost (USD)": 500.0, 
        "Processing Cost (USD)": 200.0, 
        "Emissions to Air CO2 (kg)": 3081.47509765625, 
        "Emissions to Air SOx (kg)": 23.762849807739258, 
        "Emissions to Air NOx (kg)": 19.012903213500977, 
        "Emissions to Air Particulate Matter (kg)": 11.879419326782227, 
        "Emissions to Water Acid Mine Drainage (kg)": 5.273314476013184, 
        "Emissions to Water Heavy Metals (kg)": 3.1632373332977295, 
        "Emissions to Water BOD (kg)": 2.1091058254241943, 
        "Greenhouse Gas Emissions (kg CO2-eq)": 5035.37841796875, 
        "Scope 1 Emissions (kg CO2-eq)": 2504.817626953125, 
        "Scope 2 Emissions (kg CO2-eq)": 1503.769775390625, 
        "Scope 3 Emissions (kg CO2-eq)": 1044.5374755859375, 
        "End-of-Life Treatment": "Recycling", 
        "Environmental Impact Score": 56.69657897949219, 
        "Metal Recyclability Factor": 0.5509214401245117, 
        "Energy_per_Material": 11.394341468811035, 
        "Total_Air_Emissions": 237.7609100341797, 
        "Total_Water_Emissions": 10.54836368560791, 
        "Transport_Intensity": 8.591459274291992, 
        "GHG_per_Material": 5.105621337890625, 
        "Time_Period_Numeric": 2017.448974609375, 
        "Total_Cost": 780.7816162109375, 
        "Circularity_Score": 44.7951545715332, 
        "Circular_Economy_Index": 0.46311068534851074, 
        "Recycled Content (%)": 19.121646881103516, 
        "Resource Efficiency (%)": 19.40056610107422, 
        "Extended Product Life (years)": 22.87418556213379, 
        "Recovery Rate (%)": 87.98934936523438, 
        "Reuse Potential (%)": 29.665084838867188 
    }

    result = make_prediction(sample_input)
    import json
    print("\nPrediction Results:")
    print(json.dumps(result, indent=2))
