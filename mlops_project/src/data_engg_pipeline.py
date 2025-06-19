import os
from src.data_analysis import analyze_data, load_and_combine_data
from src.data_preperation import prepare_data_encoding
from src.feature_enginnering import prepare_data_for_feast
from src.data_validation import validate_data
import pandas as pd

def data_pipeline():
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

    # 3. Data Preparation (encodings)
    output_parquet_path = os.path.join(data_dir, '../', 'feature_store/data/', 'employee_preprocessed_data.parquet')
    final_df = prepare_data_encoding(validated_employee_data)

    # 4. FeatureEnginnering
    prepare_data_for_feast(final_df, validated_employee_data, output_parquet_path)


if __name__ == "__main__":
    data_pipeline()