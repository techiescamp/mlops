from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import pandas as pd
import pickle

import mlflow
import mlflow.pyfunc


app = Flask(__name__)
CORS(app)

# mlflow
mlflow.set_tracking_uri("http://127.0.0.1:5000")

model_name = "Employee Attrition Model"
model_version = "3"
run_id = "bdda2dfd55454b9694bef6653ebbbe64"
model = mlflow.pyfunc.load_model(f"models:/{model_name}/{model_version}")

# download artifacts
scaler_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/scaler.pkl")
with open(scaler_path, "rb") as f:
    scaler = pickle.load(f)

feature_names_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/feature_names.pkl")
with open(feature_names_path, "rb") as f:
    feature_names = pickle.load(f)

ordinal_encoder_path = mlflow.artifacts.download_artifacts(f"runs:/{run_id}/preprocessor/ordinal_encoder.pkl")
with open(ordinal_encoder_path, "rb") as f:
    ordinal_encoder = pickle.load(f)

def preprocessing_input(input):
    # ordinal encoding
    cols_to_encode = ['Work-Life Balance', 'Job Satisfaction', 'Performance Rating', 'Education Level', 'Job Level', 'Company Size', 'Company Reputation', 'Employee Recognition']
    input[cols_to_encode] = ordinal_encoder.transform(input[cols_to_encode]).astype('int')

    # binary encoding
    binary_cols = ['Overtime', 'Remote Work', 'Opportunities']
    for col in binary_cols:
        input[col] = input[col].map({'No': 0, 'Yes': 1})
    
    # feature engg
    def map_monthly_income(income):
        if 1 <= income <= 10000:
            return 0
        elif 10001 <= income <= 20000:
            return 1
        elif 20001 <= income <= 50000:
            return 2
        elif 50001 <= income <=100000:
            return 3
        elif income >= 100001:
            return 4
        else:
            return -1
    input['Monthly Income'] = input['Monthly Income'].apply(map_monthly_income)

    # ensure correct column order
    input = input[feature_names]
    print(input)

    # scale the data
    input_scaled = scaler.transform(input)
    return input_scaled


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()
    if isinstance(data, dict):
        input_data = pd.DataFrame([data])
    else:
        input_data = pd.DataFrame(data)
    print('data', input_data)

    df = preprocessing_input(input_data)
    print('df: ', df)

    try:
        prediction = model.predict(df)
        print('predict: ', prediction)

        result = "Left" if prediction[0] == 1 else "Stayed"
        print('result: ', result)
        return jsonify({"prediction": result})
    except Exception as e:
        return jsonify({
            "error": str(e)
        }), 400


if __name__ == "__main__":
    app.run(host='127.0.0.1', port=8000, debug=True)