from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


def model_evaluation(model, X_test, y_test):

    # Make predictions
    y_pred = model.predict(X_test)

    # Calculate metrics
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred)
    }

    return metrics, y_pred
    