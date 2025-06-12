import os
from src.data_analysis import analyze_data, load_and_combine_data
from src.data_preperation import prepare_data_for_feast
from src.train_model import train_attrition_model
from src.data_validation import validate_data
import pandas as pd
import datetime

def run_pipeline():
    data_dir = os.path.dirname(__file__)

    # 1. Data Loading and Analysis
    employee_data, combined_data_path = load_and_combine_data(data_dir)
    if os.path.exists(combined_data_path) is False:
        print("Pipeline aborted: Data loading failed.")
        return
    
    analyze_data(employee_data)

    # 2. Data Validation
    validated_employee_data = validate_data(employee_data)
    if validated_employee_data is None:
        print("Pipeline aborted: Data validation failed.")
        return


    # 3. Data Preparation for Feast
    output_parquet_path = os.path.join(data_dir, '..', 'feature_store/data', 'employee_preprocessed_data.parquet')

    # Create unique employee_id and timestamp for Feast
    validated_employee_data["employee_id"] = validated_employee_data.index + 1
    validated_employee_data['event_timestamp'] = pd.to_datetime(datetime.datetime.now()) - pd.to_timedelta(validated_employee_data.index, unit='D')
    print("Added 'employee_id' and 'event_timestamp' for feast")

    X = validated_employee_data.drop(['Employee ID', 'Attrition', 'Job Role', 'Distance from Home', 'Marital Status', 'Gender'], axis=1)
    y = validated_employee_data['Attrition']
    prepare_data_for_feast(X, y, output_parquet_path)

    # 4. Model Training
    train_attrition_model()

if __name__ == "__main__":
    run_pipeline()