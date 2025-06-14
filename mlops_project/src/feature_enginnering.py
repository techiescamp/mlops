import datetime
import os
import pandas as pd
from feast import FeatureStore
from feature_store.features import employee_features_fv


def prepare_data_for_feast(final_df: pd.DataFrame, raw_data: pd.DataFrame, output_parquet_path: str):
    # Create unique employee_id and timestamp for Feast
    final_df = final_df.copy()
    raw_data = raw_data.copy()

    if "Employee ID" not in raw_data.columns:
        raise ValueError("'Employee ID' column not found in raw_data.")
    else:
        final_df['employee_id'] = raw_data["Employee ID"].values

    final_df['event_timestamp'] = pd.to_datetime(datetime.datetime.now()) - pd.to_timedelta(final_df.index, unit='D')
    print("Added 'event_timestamp' for feast")

    # ensoure output_data directory exists !
    output_dir = os.path.dirname(output_parquet_path)
    print(output_dir)
    
    # Create the directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # If file exists, delete it
    if os.path.exists(output_parquet_path):
        try:
            os.remove(output_parquet_path)
            print(f"Deleted existing file at: {output_parquet_path}")
        except PermissionError as e:
            raise PermissionError(f"Cannot overwrite file. It's likely open or locked.\nDetails: {e}")

    
    # save to parquet
    final_df.to_parquet(output_parquet_path, index=False)

    print("Data preparation complete and saved successfully.")
    print(f"Final data columns: {final_df.columns.tolist()}")
    print("Column names: ", {final_df.shape})



def get_training_data_from_feast():
    script_dir = os.path.dirname(__file__)
    feast_repo_path = os.path.join(script_dir, "../feature_store")

    # import feast features
    MODEL_INPUT_FEATURE_ORDER = sorted([
        field.name for field in employee_features_fv.schema
        if field.name not in ["employee_id", "attrition_label", "event_timestamp", "created_timestamp"]
    ])


    fs = FeatureStore(repo_path=feast_repo_path)
    preprocessed_df_path = os.path.join(script_dir, '../feature_store/data', 'employee_preprocessed_data.parquet')
    if not os.path.exists(preprocessed_df_path):
        print(f"Preprocessed data not found at: {preprocessed_df_path}")
        return None
    
    entity_df = pd.read_parquet(preprocessed_df_path, columns=['employee_id', 'event_timestamp'])
    all_features_to_fetch_from_feast = [f"employee_preprocessed_features:{feature}" for feature in MODEL_INPUT_FEATURE_ORDER]
    all_features_to_fetch_from_feast.append(f"employee_preprocessed_features:attrition_label")


    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=all_features_to_fetch_from_feast
    ).to_df()
    print("training_df: ", training_df)

    columns_to_drop_from_training_df = [
        col for col in training_df.columns
        if col.startswith(('employee_id', 'event_timestamp', 'created_timestamp'))
    ]
    model_features = training_df.drop(columns_to_drop_from_training_df, axis=1)
    print(f"Training data columns: {training_df.columns.tolist()}")
    print(f"Model Features columns: {model_features.columns.tolist()}")

    return model_features

  
