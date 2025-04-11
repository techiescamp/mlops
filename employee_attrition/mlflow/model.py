from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def train_model(x_train_scaled, x_test_scaled, y_train, y_test):
    lr = LogisticRegression(random_state=42)
    lr.fit(x_train_scaled, y_train)

    # predict
    y_pred = lr.predict(x_test_scaled)
    # metrics evaluation
    accuracy = accuracy_score(y_pred, y_test)

    return lr, accuracy
