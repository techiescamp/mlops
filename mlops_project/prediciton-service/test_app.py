import requests
import json

FEAST_SERVER_URL = "http://localhost:6566" # Or the load balancer URL if on K8s

def get_employee_features_via_server(employee_ids: list):
    payload = {
        "features": [
            "employee_preprocessed_features:Age",
            "employee_preprocessed_features:Company Reputation",
            "employee_preprocessed_features:Company Size",
            "employee_preprocessed_features:Company Tenure",
            "employee_preprocessed_features:Education Level",
            "employee_preprocessed_features:Employee Recognition",
            "employee_preprocessed_features:Job Level",
            "employee_preprocessed_features:Job Satisfaction",
            "employee_preprocessed_features:Monthly Income",
            "employee_preprocessed_features:Number of Dependents",
            "employee_preprocessed_features:Number of Promotions",
            "employee_preprocessed_features:Opportunities",
            "employee_preprocessed_features:Overtime",
            "employee_preprocessed_features:Performance Rating",
            "employee_preprocessed_features:Remote Work",
            "employee_preprocessed_features:Work-Life Balance",
            "employee_preprocessed_features:Years at Company",
            # Do NOT include "employee_preprocessed_features:attrition_label" for online inference
        ],
        "entities": {
            "employee_id": employee_ids
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
        return sorted(feature_names)
    except requests.exceptions.RequestException as e:
        print(f"Error communicating with Feast server: {e}")
        return None


if __name__ == "__main__":
    employee_ids_to_fetch = [8410]
    features_data = get_employee_features_via_server(employee_ids_to_fetch)
    if features_data:
        print(features_data)
        # You would then process features_data into a format suitable for your model