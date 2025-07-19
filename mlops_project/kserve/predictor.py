import pandas as pd
import mlflow
from kserve import Model, ModelServer
import os
from dotenv import load_dotenv

load_dotenv()

# MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_ARTIFACT_URL", "../mlruns/0/models/m-3eebf8d27f0c4f039b54753013dd19bf/artifacts")
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://localhost:5000")
MODEL_NAME = "Employee Attrition Model"
STAGE = "Production"

mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

class EmployeeAttritionPrediction(Model):
    def __init__(self, name, model_uri):
        super().__init__(name)
        self.model_uri = model_uri
        self.ready = False

    def load(self):
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            self.model = mlflow.sklearn.load_model(self.model_uri)
            self.ready = True
        except Exception as e:
            print(f"Error during load: {e}")
            self.ready = False


    def predict(self, payload, headers=None):
        print(f"Recieved payload: {payload}")

        instances = payload.get("instances", [])
        if not instances:
            return {"error": "No instances provided."}

        df = pd.DataFrame(instances)
        
        print(f"Input DataFrame shape: {df.shape}")
        print(f"Input DataFrame columns: {df}")
        
        # Predict using MLflow model
        predictions = self.model.predict(df)
        prediction_proba = self.model.predict_proba(df)[0][1]
        print(f"prediction: {predictions}")
        print(f"prediction_proba: {prediction_proba}")

        # Return results as list
        return {
            "predictions": predictions.tolist(), 
            "prediction_proba": prediction_proba.tolist()
        }


if __name__ == "__main__":
    #  construct model uri from registry
    model_uri = f"models:/{MODEL_NAME}/{STAGE}"
    print(f"Using Mlflow tracking URI: {MLFLOW_TRACKING_URI}")
    print(f"Using model registry uri: {model_uri}")

    server = ModelServer(http_port=8002)
    print(f"mlflow-url in kserve: {os.environ.get("MLFLOW_ARTIFACT_URL")}")
    model = EmployeeAttritionPrediction(
        name="mlops_employee_attrition",
        model_uri=model_uri
    )
    model.load()
    server.start(models=[model])

