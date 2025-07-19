import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature
import os
from dotenv import load_dotenv

load_dotenv()


def model_registry(model, X_train, metrics):
    # Start MLflow run
    mlflow_tracking_uri = os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
    mlflow.set_tracking_uri(mlflow_tracking_uri)


    with mlflow.start_run(run_name="employee_attrition_run") as run:
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        print("Logged metrics")

        # log model
        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="attrition_model_pipeline",
            registered_model_name="Employee Attrition Model", # Register the model in MLflow Model Registry
            signature=infer_signature(X_train), # mlflow.models.signature.
            input_example=X_train
        )
        # register model
        print(f"You can view the run details at: {run.info.run_id}")

