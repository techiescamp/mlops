import pandas as pd
import mlflow
from kserve import Model, ModelServer
from feast import FeatureStore
import os
from dotenv import load_dotenv

load_dotenv()

class EmployeeAttritionPrediction(Model):
    def __init__(self, name, model_uri, feast_repo_path):
        super().__init__(name)
        self.model_uri = model_uri
        self.feast_repo_path = feast_repo_path
        self.ready = False

    def load(self):
        try:
            self.model = mlflow.sklearn.load_model(self.model_uri)
            self.feast_store = FeatureStore(repo_path=self.feast_repo_path)
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

        if 'employee_id' not in df.columns:
            return {"error": "'employee_id' is required in the input"}

        # Get preprocessed features from Feast
        feast_features = self.feast_store.get_online_features(
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
                "employee_preprocessed_features:Years at Company"    
            ],
            entity_rows=df[["employee_id"]].to_dict(orient="records")
        ).to_df()

        print(f"features length: {len(feast_features.columns)}")
        sorted_features = sorted([
            col for col in feast_features.columns
            if col not in ["employee_id", "attrition_label", "event_timestamp", "created_timestamp"]
        ])
        # Drop employee_id from final features and reorder the feature
        X = df.drop(columns=["employee_id"], axis=1)[sorted_features]
        
        print(f"Input DataFrame shape: {X.shape}")
        print(f"Input DataFrame columns: {list(X.columns)}")
        
        # Predict using MLflow model
        predictions = self.model.predict(X)
        print(f"prediction: {predictions}")

        # Return results as list
        return {"predictions": predictions.tolist()}


if __name__ == "__main__":
    server = ModelServer(http_port=8002)
    model = EmployeeAttritionPrediction(
        name="mlops_employee_attrition",
        model_uri=os.environ["MLFLOW_ARTIFACT_URL"],
        feast_repo_path="../feature_store"
    )
    model.load()
    server.start(models=[model])

