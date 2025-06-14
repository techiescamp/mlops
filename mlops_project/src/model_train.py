# train_model.py
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def model_training(X_train, y_train):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('logistic_regression', LogisticRegression(random_state=42))
    ])
    model_pipeline.fit(X_train, y_train)
    print("Model pipeline trained.")

    return model_pipeline
    
if __name__ == "__main__":
   print(" --- model training ---")

