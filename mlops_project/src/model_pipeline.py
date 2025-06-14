import os
import pandas as pd
from sklearn.model_selection import train_test_split
from src.feature_enginnering import get_training_data_from_feast
from src.model_train import model_training
from src.model_evaluation import model_evaluation
from src.model_validation import model_validation
from src.model_registry import model_registry

def run_pipeline():
    # 1. get features from feast
    df = get_training_data_from_feast()

    # Split data
    X = df.drop(columns=["attrition_label"], axis=1)
    y = df["attrition_label"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


    # 2. Model Training
    model = model_training(X_train, y_train)

    # 3. Model Evaluation
    metrics, y_pred = model_evaluation(model, X_test, y_test)
    print(f"Metrics: {metrics}")

    # 4. Model Validation
    if model_validation(metrics):
        model_registry(model, X_train, metrics)
    else:
        print("Model rejected and not registered.")


if __name__ == "__main__":
    run_pipeline()