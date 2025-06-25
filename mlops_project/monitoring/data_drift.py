import pandas as pd
import requests
import json
import os
from scipy.stats import ks_2samp, chi2_contingency


alert_url="http://localhost:8001/alert"


# Load historical and current data
reference_data = pd.read_parquet("../feature_store/data/employee_preprocessed_data.parquet")
live_data = pd.read_csv("../raw_data/input_data.csv")

ref_data = reference_data.drop(columns=['employee_id', 'event_timestamp', 'attrition_label'], errors='ignore')
curr_data = live_data.drop(columns=['employee_id'], errors='ignore')

categorical_features = [
    'Age', 'Monthly Income', 'Work-Life Balance', 'Job Satisfaction', 'Performance Rating',
    'Education Level', 'Job Level', 'Company Size', 'Company Reputation', 'Employee Recognition'
]

numerical_features = ['Overtime', 'Remote Work', 'Opportunities', 'Years at Company', 'Number of Promotions', 'Number of Dependents', 'Company Tenure']

# Columns to check (numeric + encoded categorical)
drift_results = {}

for col in ref_data.columns:
    if col in numerical_features:  # numerical
        stat, p_value = ks_2samp(ref_data[col], curr_data[col])
        drift_results[col] = {
            "type": "numerical",
            "stat": stat,
            "p_value": p_value, 
            "drift": p_value < 0.05,
        }
    elif col in categorical_features:  # categorical
        # Create contingency table
        contingency_table = pd.crosstab(ref_data[col], curr_data[col])
        stat, p_value, _, _ = chi2_contingency(contingency_table)
        drift_results[col] = {
            "type": "categorical",
            "stat": stat,
            "p_value": p_value, 
            "drift": p_value < 0.05,
        }
    else:
        print(f"Skipping column {col} as it is not in the expected feature set.")
        continue

# Print results
for feature, result in drift_results.items():
    status = "⚠️ Drift detected" if result["drift"] else "✅ No drift"
    print(f"{feature} ({result['type']}): p={result['p_value']:.4f} → {status}")


#  alert if drift detected
drifted = [f for f, r in drift_results.items() if r.get('drift')]
num_drifted = len(drifted)
threshold = 5
if num_drifted >= threshold:
    print(f"ALERT: {num_drifted} features show drift: {drifted}")
    try:
        response = requests.post(
            alert_url, 
            headers={"Content-Type": "application/json"},
            data=json.dumps({
                "alert": "Data Drift detected",
                "features": drifted,
                "count": num_drifted
            }))
    except Exception as e:
        print(f"Failed to send alert: {e}")
else:
    print("No significant drift detected.")
