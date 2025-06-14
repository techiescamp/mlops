import json
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import requests
import pandas as pd
import time

# from feast import FeatureStore
# from feature_store.features import employee_features_fv

from sklearn.preprocessing import OrdinalEncoder
from pydantic import BaseModel, Field
from typing import Any, Dict, Optional
import os
from dotenv import load_dotenv

# from monitoring.logger import log_metrics


load_dotenv()

FEAST_SERVER_URL = os.environ.get("FEAST_SERVER_URL", "http://localhost:5050") # Or the load balancer URL if on K8s
KSERVE_URL = os.environ.get("KSERVE_URL", "http://localhost:8002/v1/models/mlops_employee_attrition:predict")
MONITORING_URL = os.environ.get("MONITORING_URL", "http://localhost:8001")


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


def get_employee_features_via_server(emp_id: int):
    payload = {
        "feature_service": "employee_attrition_features",
        "entities": {
            "employee_id": [emp_id]
        }
    }
    headers = {"Content-Type": "application/json"}
   
    try:
        response = requests.post(
            f"{FEAST_SERVER_URL}/get-online-features",
            data=json.dumps(payload),
            headers=headers
        )
        response.raise_for_status() # Raise an exception for HTTP errors (4xx or 5xx)
        feature_names = response.json().get('metadata', {}).get('feature_names', [])
        # Filter out employee_id as it's an entity key, not a feature for the model
        filtered_feature_names = [name for name in feature_names if name != 'employee_id']
        print(f"Feature_name: {filtered_feature_names}")
        return sorted(filtered_feature_names)
    except Exception as e:
        print(f"Error communicating with Feast server: {e}")
        return None


def log_metrics_service(data: dict):
    try:
        response = requests.post(f"{MONITORING_URL}/log", json=data)
        response.raise_for_status()
        print("Metrics sent to monitoring service successfully!")
    except requests.exceptions.RequestException as e:
        print(f"Error sending metrics to monitoring service: {e}")



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

        FINAL_MODEL_FEATURE_ORDER = get_employee_features_via_server(payload.data['employee_id'])
        final_input = preprocessed_input_data.reindex(columns=FINAL_MODEL_FEATURE_ORDER, fill_value=0)

        # validation: check for any NaNs introduced by reindexing
        if final_input is None:
            status = "input_error"
            print(f"Feast service did not find features: {final_input}")
            return {"error": "Feast service did not find features: {final_input}."}

        print(f"Final input DataFrame shape for KServe: {final_input.shape}")
        print(f"Final input DataFrame columns for KServe: {final_input.columns.tolist()}")

        # Send to KServe model running locally
        response = requests.post(KSERVE_URL, json={"instances": final_input.to_dict(orient="records")})


        response.raise_for_status() # Raise HTTPError for bad responses (4xx or 5xx)
        
        print('✅😷 result: ', response.json())
        result = response.json()
        prediction_output = result["predictions"][0]
        return {"prediction": result["predictions"][0]}
    
    except requests.exceptions.RequestException as e:
        status = "kserve_error"
        print(f"Error communicating with KServe: {e}")
        raise HTTPException(status_code=500, detail=f"Error from KServe: {e}. Check KServe logs for details.")

    except Exception as e:
        status = "internal_error"
        return {"error": str(e)}
    
    finally:
        end_time = time.time()
        latency = (end_time - start_time) * 1000
        log_metrics_service({
            "latency_ms": latency,
            "status": status,
            "prediction": prediction_output,
        })



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
