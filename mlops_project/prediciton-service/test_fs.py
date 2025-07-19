import json
import requests


<<<<<<< HEAD
FEAST_SERVER_URL = "http://localhost:5050"
# FEAST_SERVER_URL = "http://4.154.210.230:30800"
=======
# FEAST_SERVER_URL = "http://localhost:5050"
FEAST_SERVER_URL = "http://4.154.210.230:30800"

>>>>>>> 802631f5eca657d7ec6984c1ef9a4aeca3d47f57


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
    # feature_names = response.json().get('metadata', {}).get('feature_names', [])
    feature_names = response.json()
    print(f"Feature_name: {feature_names}")
except Exception as e:
     print(f"error: {e}")
