
def model_validation(metrics, thresholds={"f1_score": 0.65}):
    for metric, threshold in thresholds.items():
        if metrics[metric] < threshold:
            print(f"Model failed validation: {metric} = {metrics[metric]} < {threshold}")
            return False
        print("Model passed validation")
        return True