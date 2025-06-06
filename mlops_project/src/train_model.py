# train_model.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from feast import FeatureStore
import logging
import os

logging.basicConfig(level=logging.INFO)
# logger = logging.get# logger(__name__)

# --- MLflow Tracking URI Setup ---
# Set the MLflow tracking URI. If not set, MLflow defaults to a local 'mlruns' directory.
# For local testing, you can use 'http://localhost:5000' if you run 'mlflow ui' separately.
# For production setups, this would typically point to a remote MLflow Tracking Server.
mlflow_tracking_uri = "http://127.0.0.1:5000" # Using 127.0.0.1 instead of localhost for broader compatibility
mlflow.set_tracking_uri(mlflow_tracking_uri)
# logger.info(f"MLflow Tracking URI set to: {mlflow_tracking_uri}")
# --- End MLflow Tracking URI Setup ---


# Initialize Feast FeatureStore
# Assumes feature_store.yaml and features.py are in the current directory
script_dir = os.path.dirname(__file__)
feast_repo_path = os.path.abspath(os.path.join(script_dir, "../feature_store"))
print(f"------------------- {feast_repo_path} ------------------")
print(feast_repo_path)
fs = FeatureStore(repo_path=feast_repo_path)
# logger.info(f"Feast FeatureStore initialized with repo_path: {feast_repo_path}")
print(f"✅ Feast FeatureStore initialized with repo_path: {feast_repo_path}")

# Define your features and label names as they appear in the preprocessed data
# Ensure these match the 'schema' defined in features.py
feature_names = [
    "Work-Life Balance", "Job Satisfaction", "Performance Rating",
    "Education Level", "Job Level", "Company Size", "Company Reputation",
    "Employee Recognition", "Overtime", "Remote Work",
    "Monthly Income", "Opportunities"
]
label_name = "attrition_label"

def get_training_data_from_feast():
    """
    Retrieves historical feature data and the attrition label from Feast.
    """
    # logger.info("Getting training data from Feast...")

    # We need an entity DataFrame with 'employee_id' and 'event_timestamp'to query historical features from Feast.
    # In a real scenario, this would come from your data warehouse,
    # representing the historical points for which you want to fetch features.
    # For this example, we'll load the employee_id and event_timestamp from the
    # preprocessed parquet file itself.
    preprocessed_df_path = os.path.join(script_dir, '../feature_store/data', 'employee_preprocessed_data.parquet')
    if not os.path.exists(preprocessed_df_path):
        print(f"✅ Preprocessed data not found at: {preprocessed_df_path}")
        print("Please run data_preparation.py first to create the data.")
        # logger.error(f"Preprocessed data not found at: {preprocessed_df_path}")
        # logger.error("Please run data_preparation.py first to create the data.")
        return None

    # Load only necessary columns to create the entity_df
    entity_df = pd.read_parquet(preprocessed_df_path, columns=['employee_id', 'event_timestamp', 'attrition_label'])

    # Request features from the defined feature view
    # Note: We are fetching the attrition_label as well since it's in our source data
    # and defined in the feature view.
    # Construct the list of features to fetch, including the label for training
    all_features_to_fetch = [f"employee_preprocessed_features:{feature}" for feature in feature_names]
    all_features_to_fetch.append(f"employee_preprocessed_features:{label_name}") # Add the label as well

    training_data_with_label = fs.get_historical_features(
        entity_df=entity_df[['employee_id', 'event_timestamp']], # Only entities for fetching features
        features=all_features_to_fetch,
    ).to_df()

    # The 'attrition_label' will be part of the fetched data if it's in the source
    # and defined in the FeatureView schema.
    if label_name not in training_data_with_label.columns:
        # logger.error(f"Error: '{label_name}' not found in fetched historical features.")
        print(f"Error: '{label_name}' not found in fetched historical features.")
        return None

    # logger.info(f"Training data shape after Feast retrieval: {training_data_with_label.shape}")
    # logger.info(f"Training data columns: {training_data_with_label.columns.tolist()}")
    print(f"Training data shape after Feast retrieval: {training_data_with_label.shape}")
    print(f"✅ Training data columns: {training_data_with_label.columns.tolist()}")


    # Drop Feast-specific metadata columns if present and not needed for training
    # Also ensure 'employee_id' itself is dropped from features if it's not a feature
    columns_to_drop_from_training_data = [
        col for col in training_data_with_label.columns
        if col.startswith(('employee_id', 'event_timestamp', 'created_timestamp'))
    ]
    # Ensure attrition_label is also not in features
    if label_name in columns_to_drop_from_training_data:
        columns_to_drop_from_training_data.remove(label_name)

    training_data_with_label = training_data_with_label.drop(columns=columns_to_drop_from_training_data, errors='ignore')

    return training_data_with_label




def train_attrition_model():
    """
    Trains an employee attrition prediction model using data from Feast,
    logs metrics and the model with MLflow.
    """
    df = get_training_data_from_feast()
    if df is None:
        # logger.error("Failed to get training data. Exiting.")
        print("Failed to get training data. Exiting.")
        return

    # Separate features (X) and label (y)
    # Ensure only the actual features are in X and the label is in y
    X = df[feature_names]
    y = df[label_name]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    # logger.info(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    # Start MLflow run
    with mlflow.start_run(run_name="mlops_employee_attrition") as run:
        # Get current MLflow Run ID
        run_id = run.info.run_id
        # logger.info(f"MLflow Run ID: {run_id}")
        print(f"MLflow Run ID: {run_id}")

        # Define model parameters
        solver = "liblinear"
        C = 0.1 # Regularization parameter

        # Log parameters
        mlflow.log_param("solver", solver)
        mlflow.log_param("C", C)
        # logger.info(f"Logged parameters: solver={solver}, C={C}")
        print(f"✅ Logged parameters: solver={solver}, C={C}")

        # Create a scikit-learn Pipeline that includes scaling and the model
        # This is crucial for serving, as it packages preprocessing with the model.
        pipeline = Pipeline([
            ('scaler', StandardScaler()),
            ('logistic_regression', LogisticRegression(solver=solver, C=C, random_state=42))
        ])

        # Train the pipeline
        # logger.info("Training the model pipeline...")
        print("Training the model pipeline...")

        pipeline.fit(X_train, y_train)
        # logger.info("Model pipeline trained.")
        print("Model pipeline trained.")

        # Make predictions
        y_pred = pipeline.predict(X_test)
        y_pred_proba = pipeline.predict_proba(X_test)[:, 1] # Probability of attrition (class 1)

        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        # logger.info(f"Logged metrics: Accuracy={accuracy:.4f}, Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")
        print(f"✅ Logged metrics: Accuracy={accuracy:.4f},\n Precision={precision:.4f},\n Recall={recall:.4f},\n F1-Score={f1:.4f}\n")

        signature = infer_signature(X_train, y_pred) # mlflow.models.signature.

        # Log the scikit-learn pipeline as an MLflow artifact
        # This will save the scaler and the logistic regression model together.
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="attrition_model_pipeline",
            registered_model_name="Employee Attrition Model", # Register the model in MLflow Model Registry
            signature=signature,
            input_example=X_train
        )
        # logger.info("Model pipeline logged to MLflow Model Registry as 'AttritionPredictionModel'.")
        # logger.info(f"MLflow UI can be accessed by running 'mlflow ui' in your terminal.")
        # logger.info(f"You can view the run details at: mlflow/#/runs/{run_id}")

        print("Model pipeline logged to MLflow Model Registry as 'AttritionPredictionModel'.")
        print(f"MLflow UI can be accessed by running 'mlflow ui' in your terminal.")
        print(f"✅ You can view the run details at: mlflow/#/runs/{run_id}")

if __name__ == "__main__":
    # Ensure data/ directory exists for Feast setup
    if not os.path.exists('../feature_store/data'):
        print("++++++ wrong-path ++++++++++")
        # os.makedirs('data')

    # Run the training process
    train_attrition_model()

