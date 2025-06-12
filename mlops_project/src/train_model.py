# train_model.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
from feast import FeatureStore
from feature_store.features import employee_features_fv
import os
from dotenv import load_dotenv


script_dir = os.path.dirname(__file__)
dotenv_path = os.path.join(script_dir, '../', '.env')
load_dotenv(dotenv_path=dotenv_path)

# import feast features
MODEL_INPUT_FEATURE_ORDER = sorted([
    field.name for field in employee_features_fv.schema
    if field.name not in ["employee_id", "attrition_label", "event_timestamp", "created_timestamp"]
])
feast_repo_path = os.path.join(script_dir, "../feature_store")
print(f"Feast Repo Path: {feast_repo_path}")

def get_training_data_from_feast():
    fs = FeatureStore(repo_path=feast_repo_path)
    preprocessed_df_path = os.path.join(script_dir, '../feature_store/data', 'employee_preprocessed_data.parquet')
    if not os.path.exists(preprocessed_df_path):
        print(f"Preprocessed data not found at: {preprocessed_df_path}")
        print("Please run data_preparation.py first to create the data.")
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



def train_attrition_model():
    df = get_training_data_from_feast()
    
    X = df.drop(columns=["attrition_label"], axis=1)
    y = df["attrition_label"]

    print("y: ", y.head())

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    print("Model pipeline trained.")

    # Make predictions
    y_pred = pipeline.predict(X_test)
        
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Start MLflow run
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)


    with mlflow.start_run(run_name="mlops_employee_attrition") as run:
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print(f"Logged metrics: Accuracy={accuracy:.4f},\n Precision={precision:.4f},\n Recall={recall:.4f},\n F1-Score={f1:.4f}\n")

        signature = infer_signature(X_train, y_pred) # mlflow.models.signature.

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="attrition_model_pipeline",
            registered_model_name="Employee Attrition Model", # Register the model in MLflow Model Registry
            signature=signature,
            input_example=X_train,
        )
        mlflow.register_model(f"runs:/{run.info.run_id}/model", "Employee_Attrition_Model")
        print(f"You can view the run details at: mlflow/#/runs/{run.info.run_id}")


if __name__ == "__main__":
    train_attrition_model()

