from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import pandas as pd
import time

from feast import FeatureStore
from feature_store.features import employee_features_fv

from sklearn.preprocessing import OrdinalEncoder
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import os
from dotenv import load_dotenv

from monitoring.logger import log_metrics


load_dotenv()

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)


# ==== Define encoder logic same as your data-preparation.py ====
encoder_columns = [
    'Work-Life Balance', 'Job Satisfaction', 'Performance Rating',
    'Education Level', 'Job Level', 'Company Size',
    'Company Reputation', 'Employee Recognition'
]
encoder_categories = [
    ['Poor', 'Fair', 'Good', 'Excellent'],
    ['Low', 'Medium', 'High', 'Very High'],
    ['Low', 'Below Average', 'Average', 'High'],
    ["High School", "Bachelor’s Degree", "Master’s Degree", "Associate Degree", "PhD"],
    ['Entry', 'Mid', 'Senior'],
    ['Small', 'Medium', 'Large'],
    ['Poor', 'Fair', 'Good', 'Excellent'],
    ['Low', 'Medium', 'High', 'Very High'],
]

oe = OrdinalEncoder(categories=encoder_categories, handle_unknown='use_encoded_value', unknown_value=-1)

# Fit encoder with dummy rows using all categories
# Pad category lists to the same length
max_len = max(len(cat) for cat in encoder_categories)
padded_categories = [
    cat + [cat[-1]] * (max_len - len(cat))  # Repeat last element to match max length
    for cat in encoder_categories
]

# Now zip into rows
dummy_rows = list(zip(*padded_categories))
print('dummy rows: ', dummy_rows)
dummy_data = pd.DataFrame.from_records(dummy_rows, columns=encoder_columns)
print('dummy data: ', dummy_data)
oe.fit(dummy_data)


def preprocess_input(data: dict):
    df = pd.DataFrame([data])
    
    # Boolean mapping
    bool_cols_map = {
        'Overtime': {'No': 0, 'Yes': 1},
        'Remote Work': {'No': 0, 'Yes': 1},
        'Opportunities': {'No': 0, 'Yes': 1}
    }
    for col, mapping in bool_cols_map.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(-1) # Fillna for robustness if category missing

    # Map Monthly Income
    def map_income(income):
        income = float(income)
        if 1200 <= income <= 10000: return 0
        elif 10001 <= income <= 20000: return 1
        elif 20001 <= income <= 35000: return 2
        elif 35001 <= income <= 50000: return 3
        elif income >= 50001: return 4
        else: return -1

    if 'Monthly Income' in df:
        df['Monthly Income'] = df['Monthly Income'].apply(map_income)

    # Apply Ordinal Encoding
    for col in encoder_columns:
        if col not in df.columns:
            print(f"⚠️ Column '{col}' expected for ordinal encoding not found. Adding with default -1.")
            df[col] = -1 # Add missing column with a default encoded value
    df[encoder_columns] = oe.transform(df[encoder_columns]).astype('int')
    print(f"df length: {len(df.columns)}")
    return df


def sort_features():
    feast_repo_path = "./feature_store"
    FEAST_STORE = FeatureStore(repo_path=feast_repo_path)
    print(f"Feast FeatureStore initialized with repo_path: {FEAST_STORE.repo_path}")

    # --- Dynamic Model Feature List Generation (Canonical Order) ---
    # This list MUST exactly match the features and their order used for model training (X_train)
    # and the features KServe's predictor expects.
    final_model_feature_order = sorted([
        field.name for field in employee_features_fv.schema
        if field.name not in ["employee_id", "attrition_label", "event_timestamp", "created_timestamp"]
    ])
    print(f"Final model feature order: {final_model_feature_order}")
    return final_model_feature_order


class FormData(BaseModel):
    data: Dict[str, Any]


@app.post("/predict")
async def predict(payload: FormData):
    # for monitoring
    start_time = time.time()
    prediction_output = None
    status = "success"
        
    try:
        print("✅ Prediciton service called ------")
        print(f"{payload}")

        preprocessed_input_data = preprocess_input(payload.data)
        print(f"✅Transformed Input: ", preprocessed_input_data.columns)

        FINAL_MODEL_FEATURE_ORDER = sort_features()
        final_input = preprocessed_input_data.reindex(columns=FINAL_MODEL_FEATURE_ORDER, fill_value=0)

        # validation: check for any NaNs introduced by reindexing
        if final_input.isnull().any().any():
            status = "input_error"
            print(f"❌ NaNs found in final model input DataFrame after reindexing: {final_input.isnull().sum().to_dict()}")
            return {"error": "Missing or invalid features after preprocessing. Check input data and feature definitions."}

        print(f"Final input DataFrame shape for KServe: {final_input.shape}")
        print(f"Final input DataFrame columns for KServe: {final_input.columns.tolist()}")

        # Send to KServe model running locally
        kserve_url = os.environ.get("KSERVE_URL", "http://localhost:8002/v1/models/mlops_employee_attrition:predict")
        response = requests.post(kserve_url, json={"instances": final_input.to_dict(orient="records")})
        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        print('✅😷 result: ', response.json())
        result = response.json()
        prediction_output = result["predictions"][0]
        return {"prediction": result["predictions"][0]}
    
    except requests.exceptions.RequestException as e:
        status = "kserve_error"
        print(f"Error communicating with KServe: {e}", exc_info=True)
        return {"error": f"Error from KServe: {e}. Check KServe logs for details."}

    except Exception as e:
        status = "internal_error"
        return {"error": str(e)}
    
    finally:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        log_metrics({
            "latency_ms": latency,
            "status": status,
            "prediction": prediction_output,
        })

@app.get("/metrics")
async def metrics():
    import csv
    csv_path = os.path.join("monitoring", "inference_logs.csv")
    
    if not os.path.exists(csv_path):
        return {"error": "metrics file not found"}
    
    with open(csv_path, newline='') as file:
        reader = csv.DictReader(file)
        logs = list(reader)

    return {"metrics": logs}
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
