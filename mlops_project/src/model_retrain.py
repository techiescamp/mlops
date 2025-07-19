import pandas as pd
from src.model_pipeline import run_pipeline

historical_data = pd.read_parquet("../feature_store/data/employee_preprocessed_data.parquet")

new_data = pd.read_csv("../raw_data/input_data.csv")

new_data["attrition_label"] = pd.read_csv("../raw_data/prediction_output.csv")["attrition_label"]

updated_dataset = pd.concat([historical_data, new_data], ignore_index=True)

X = updated_dataset.drop(columns=['employee_id', 'event_timestamp', 'attrition_label'], errors='ignore')
y = updated_dataset['attrition_label']
feature_names = X.columns.tolist()

run_pipeline(X, y, feature_names)