import pandas as pd
import requests
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os


# sart flask app
app = Flask(__name__)
CORS(app)

KSERVE_ENDPOINT = "http://52.151.18.124:30080/v1/models/employee-attrition:predict"

# home page
@app.route("/")
def index():
    return render_template('index.html')

# predict 
@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(force=True)

    # convert input to kserve expected format
    payload = {"instances": [data]} 
    print('payload: ', payload)

    # send request to kserve endpoint
    try:
        response = requests.post(KSERVE_ENDPOINT, json=payload)
        print('res: ', response)
        if response.status_code == 200:
            response_data = response.json()
            print(response_data)

            # extract predicition result
            prediction = response_data.get("predictions", [[]])[0]
            print(prediction)
            result = "Left" if prediction else "Stayed"
            print('result: ', result)
            return jsonify({"prediction": result})
        else:
            error_msg = f"KServe returned status code {response.status_code}: {response.text}"
            print(error_msg)
            return jsonify({"error": error_msg}), 500

    except Exception as e:
        error_msg = f"Exception occurred: {str(e)}"
        print(error_msg)
        return jsonify({"error": error_msg}), 500


if __name__ == "__main__":
    app.run(debug=True)