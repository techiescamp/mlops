import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

import pandas as pd
import pickle
from utils import load_emp_attr_data
from model import train_model
from sklearn.preprocessing import StandardScaler

# set mlflow tracking uri
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# set experiment name (createsone if new)
mlflow.set_experiment("Employee_Attrition_Classification")

with mlflow.start_run(run_name="employee_attrition_run") as run:
    X_train, X_test, y_train, y_test, ordinal_encoder = load_emp_attr_data()

    # save column names for later use
    column_names = X_train.columns.tolist()

    # normalize the dataset
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model, accuracy = train_model(X_train_scaled, X_test_scaled, y_train, y_test)

    # log mlfloww
    mlflow.log_metric("accuracy", accuracy * 100)
    signature = infer_signature(X_train_scaled, model.predict(X_train_scaled))

    # save model artifact
    with open("scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)
    mlflow.log_artifact("scaler.pkl", artifact_path="preprocessor")

    # save feature names artifact
    with open("feature_names.pkl", "wb") as f:
        pickle.dump(column_names, f)
    mlflow.log_artifact("feature_names.pkl", artifact_path="preprocessor")

    with open("ordinal_encoder.pkl", "wb") as f:
        pickle.dump(ordinal_encoder, f)
    mlflow.log_artifact("ordinal_encoder.pkl", artifact_path="preprocessor")

    model_info = mlflow.sklearn.log_model(
        sk_model=model,
        artifact_path="employee_attrition_model",
        signature=signature,
        input_example=X_train_scaled,
        registered_model_name="Employee Attrition Model"
    )
    print(f"Reistered Model_Uri: {model_info.model_uri}")

    print(f"Run ID: {run.info.run_id}")
    print(f"Model accuracy: {accuracy}")
    print("Model registed as: Employee Attrition Model")
