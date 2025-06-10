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
import os
from dotenv import load_dotenv


script_dir = os.path.dirname(__file__)
dotenv_path = os.path.join(script_dir, '../', '.env')
load_dotenv(dotenv_path=dotenv_path)
feast_repo_path = os.path.join(script_dir, "../feature_store")
print(f"🔍 Feast Repo Path: {feast_repo_path}")

def get_training_data_from_feast():
    fs = FeatureStore(repo_path=feast_repo_path)
    preprocessed_df_path = os.path.join(script_dir, '../feature_store/data', 'employee_preprocessed_data.parquet')
    entity_df = pd.read_parquet(preprocessed_df_path, columns=['employee_id', 'event_timestamp'])

    training_df = fs.get_historical_features(
        entity_df=entity_df,
        features=[
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
            "employee_preprocessed_features:attrition_label",
        ] 
    ).to_df()
    print("training_df: ", training_df)

    columns_to_drop_from_training_df = [
        col for col in training_df.columns
        if col.startswith(('employee_id', 'event_timestamp', 'created_timestamp'))
    ]
    model_features = training_df.drop(columns_to_drop_from_training_df, axis=1)
    print(f"✅ Training data columns: {training_df.columns.tolist()}")
    print(f"✅ Model Features columns: {model_features.columns.tolist()}")

    return model_features



def train_attrition_model():
    df = get_training_data_from_feast()
    
    X = df.drop(columns=["attrition_label"], axis=1)
    y = df["attrition_label"]

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")

    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(random_state=42))
    ])
    pipeline.fit(X_train, y_train)
    print("✅ Model pipeline trained.")

    # Make predictions
    y_pred = pipeline.predict(X_test)
        
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Start MLflow run
    mlflow_tracking_uri = os.environ["MLFLOW_TRACKING_URI"]   # http://127.0.0.1:5000
    mlflow.set_tracking_uri(mlflow_tracking_uri)


    with mlflow.start_run(run_name="mlops_employee_attrition") as run:
        # Log metrics
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)
        print(f"✅ Logged metrics: Accuracy={accuracy:.4f},\n Precision={precision:.4f},\n Recall={recall:.4f},\n F1-Score={f1:.4f}\n")

        signature = infer_signature(X_train, y_pred) # mlflow.models.signature.

        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="attrition_model_pipeline",
            registered_model_name="Employee Attrition Model", # Register the model in MLflow Model Registry
            signature=signature,
            input_example=X_train,
        )
        mlflow.register_model(f"runs:/{run.info.run_id}/model", "Employee_Attrition_Model")
        print(f"✅ You can view the run details at: mlflow/#/runs/{run.info.run_id}")


if __name__ == "__main__":
    train_attrition_model()

