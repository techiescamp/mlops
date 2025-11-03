import json
import requests
import pandas as pd


# FEAST_SERVER_URL = "http://localhost:5050"
FEAST_SERVER_URL = "http://4.246.120.68:30800"

try:
    payload = {
        "feature_service": "employee_attrition_features",
        "entities": {
            "employee_id": [8410]
        }
    }
    headers = {"Content-Type": "application/json"}
    response = requests.post(
            f"{FEAST_SERVER_URL}/get-online-features",
            data=json.dumps(payload),
            headers=headers
        )
    feature_names = response.json()['metadata']['feature_names']
    results = response.json()['results']

    values = [r['values'][0] for r in results]

    df = pd.DataFrame([values], columns=feature_names)
    print(df)
            
    print(f"Feature_name: {feature_names}")
except Exception as e:
     print(f"error: {e}")
