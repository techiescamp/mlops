from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import numpy as np
import time

def model_evaluation(model, X_train, X_test, y_test):

    # Make predictions
    start_time = time.time()
    y_pred = model.predict(X_test)
    end_time = time.time()

    y_probs = model.predict_proba(X_test)[:, 1] if hasattr(model.named_steps["classifier"], "predict_proba") else y_pred

    conf_matrix = confusion_matrix(y_test, y_pred)
    print(f"Confusion Matrix:\n{conf_matrix.tolist()}")
    
    # technical metrics
    model_metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_probs),
    }

    # operational metrics
    if len(X_test) == 0:
        raise ValueError("X_test is empty. Cannot compute latency or throughput.")
    model_latency = ((end_time - start_time) / max(len(X_test), 1)) * 1000  # in ms
    throughput = 1000 / max(model_latency, 1e-6) # requests per second
    drift = np.linalg.norm(np.mean(X_train, axis=0) - np.mean(X_test, axis=0))
    
    prediction_metrics = {
        "model_latency": model_latency,
        "throughput": throughput,   
        "data_drift": drift
    }

    # bussiness metrics
    # 1. attrition rate
    actual_attrition_rate = y_test.mean()
    predicted_attrition_rate = y_pred.mean()
    attrition_reduction = actual_attrition_rate - predicted_attrition_rate

    # 2. cost of false negatives
    #  model says "will stay" but actually "left"
    false_negatives = conf_matrix[1, 0]
    cost_per_fn = 15000
    total_cost_fn = false_negatives * cost_per_fn

    # 3. estimated ROI
    estimated_savings = (actual_attrition_rate - predicted_attrition_rate) * len(y_test) * cost_per_fn
    model_investment_cost = 5000  # Example cost of model development and deployment
    roi = (estimated_savings - model_investment_cost) / model_investment_cost * 100  # in percentage

    business_metrics = {
        "actual_attrition_rate": round(actual_attrition_rate, 4),
        "predicted_attrition_rate": round(predicted_attrition_rate, 4),
        "attrition_reduction_rate": round(attrition_reduction, 4),
        "false_negatives": int(false_negatives),
        "total_cost_false_negatives": total_cost_fn,
        "roi": round(roi, 4)
    }

    return model_metrics, prediction_metrics, business_metrics, y_pred
    