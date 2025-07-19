import mlflow
from sklearn.model_selection import train_test_split
from src.feature_enginnering import get_training_data_from_feast
from src.model_train import model_training
from src.model_evaluation import model_evaluation
from src.model_validation import model_validation
from src.model_registry import model_registry, promote_best_model_to_production
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier


def run_pipeline(X, y, feature_names):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"X_train shape: {X_train.shape}, X_test shape: {X_test.shape}")


    # 2. Model Training
    models = {
        "LogisticRegression": LogisticRegression(),
        "DecisionTreeClassifier": DecisionTreeClassifier(criterion='gini'),
        "SVC": SVC(kernel='rbf'),
        "KNeighborsClassifier": KNeighborsClassifier(n_neighbors=5),
        "RandomForestClassifier": RandomForestClassifier(n_estimators=100),
        "XGBClassifier": XGBClassifier(
            n_estimators=200,           # More trees
            learning_rate=0.05,         # Lower learning rate
            max_depth=3,                # Shallower trees generalize better
            subsample=0.8,              # Adds randomness to rows
            colsample_bytree=0.8,       # Adds randomness to features
            reg_alpha=0.1,              # L1 regularization
            reg_lambda=1.0,             # L2 regularization
            random_state=42,
            use_label_encoder=False,
            eval_metric='logloss'),
        "GradientBoostingClassifier": GradientBoostingClassifier(n_estimators=50, learning_rate=0.1, max_depth=3),
        "AdaBoostClassifier": AdaBoostClassifier(n_estimators=50, learning_rate=0.5)
    }

    for name, model in models.items():
        print(f"Training model: {name}")
        trained_model, coefficients, system_metrics = model_training(X_train, y_train, model)

        # feature importance
        if coefficients is not None:
            feature_coeffs = list(zip(feature_names, coefficients))
            top_5_features = sorted(feature_coeffs, key=lambda x: abs(x[1]), reverse=True)[:5]
            feature_importance = {feature: coeffs for feature, coeffs in top_5_features}
            print("Feature importance is logged")
        else:
            print(f"No coefficients found for {name}, skipping feature importance.")


        # 3. Model Evaluation
        metrics, prediction_metrics, bussiness_metrics, y_pred = model_evaluation(trained_model, X_train, X_test, y_test)
        print(f"✅ model evalution completed for {name} with metrics: {metrics}")


        # 4. Model Validation
        if model_validation(metrics):
            model_registry(name, trained_model, X_train, y_pred, metrics, prediction_metrics, system_metrics, bussiness_metrics, feature_importance)
        else:
            print("Model rejected and not registered.")

        print(f"Model {name} trained and logged.")

    promote_best_model_to_production()


if __name__ == "__main__":
    # 1. get features from feast
    df = get_training_data_from_feast()

    # Split data
    X = df.drop(columns=["attrition_label"], axis=1)
    y = df["attrition_label"]

    # feature names
    feature_names = X.columns

    run_pipeline(X, y, feature_names)
