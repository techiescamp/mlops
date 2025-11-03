import os
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from dotenv import load_dotenv
from mlflow.tracking import MlflowClient


load_dotenv()


def promote_best_model_to_production(model_name="Employee Attrition Model", metric="f1_score"):
    client = MlflowClient()

    # get experiment id
    experiment = mlflow.get_experiment_by_name("Employee Attrition Experiment v2")
    print(f"experiment: {experiment}")

    # find best run (highest f1_score)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=[f'metrics.{metric} DESC'],
        max_results=1
    )

    if runs.empty:
        print("No runs found")
        return
    
    best_run_id = runs.iloc[0]["run_id"]
    print(f"Best run ID: {best_run_id}")

    # 3. find model version associated with the best run
    versions = client.search_model_versions(f"name='{model_name}'")
    target_version = None
    for v in versions:
        if v.run_id == best_run_id:
            target_version = v.version
            break  

    if not target_version:
        print(f"No model version found for run ID: {best_run_id}")
        return
    
    print(f"------   Promoting version {target_version} to production  ------")

    # 4. promote model version to production
    for v in versions:
        print(v.current_stage, v.version)
        if v.current_stage == "Production":
            client.transition_model_version_stage(
                name=model_name,
                version=v.version,
                stage="Archived"
            )
            print(f"Archieved version: {v.version}")
    
    # 5. promote best model to stage
    if any(v.version == target_version and v.current_stage == "Production" for v in versions):
        print(f"Model version {target_version} is already in Production stage.")
    client.transition_model_version_stage(
        name=model_name,
        version=target_version,
        stage="Production"
    )
    print(f"Promoted model version {target_version} to Production stage.")
    
    versions = client.get_registered_model("Employee Attrition Model").latest_versions
    get_trained_model = client.get_registered_model("Employee Attrition Model")
    print(f"Model {get_trained_model} has the following versions:")

    for v in versions:
        print(f"Version {v.version} - Stage: {v.current_stage}")



def model_registry(name, model, X_train, y_pred, metrics, prediction_metrics, system_metrics, bussiness_metrics, feature_importance=None):
    # setup mlflow
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000"))
    mlflow.set_experiment("Employee Attrition Experiment v2")

    with mlflow.start_run(run_name=f"{name}_run") as run:
        # Log metrics
        for key, value in metrics.items():
            mlflow.log_metric(key, value)
        for key, value in prediction_metrics.items():
            mlflow.log_metric(key, value)
        for key, value in system_metrics.items():
            mlflow.log_metric(key, value)
        for key, value in bussiness_metrics.items():
            mlflow.log_metric(key, value)

        # Log feature importance if available
        if feature_importance:
            for feature, coeff in feature_importance.items():
                mlflow.log_metric(f"{feature}", coeff)

        # log model
        mlflow.sklearn.log_model(
            sk_model=model,
            name=name,
            registered_model_name="Employee Attrition Model", # Register the model in MLflow Model Registry
            signature=infer_signature(X_train, y_pred), # mlflow.models.signature.
            input_example=X_train.iloc[[0]] # first row as example input
        )
        # set tags
        mlflow.set_tags({
            "model_type": model.named_steps['classifier'].__class__.__name__,
            "model_version": "v3.0",
            "stage": "Training",
            "use_case": "Employee Attrition Prediction"
        })
        print(f"Model {name} registered successfully with run ID: {run.info.run_id}")


