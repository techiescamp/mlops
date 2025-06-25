import os
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
import requests

load_dotenv()

# sart flask app
app = Flask(__name__)
CORS(app)

prediction_url = os.environ.get("PREDICTION_URL")

# home page
@app.route("/", methods=['GET'])
def index(): 
    print(prediction_url)
    return render_template('index.html')


@app.route("/predict", methods=['POST'])
def predict():
    try:
        frontend_data = request.get_json()
        print(f"Frontend data : {frontend_data}")
        response = requests.post(prediction_url, json=frontend_data)
        response.raise_for_status()

        prediction_result = response.json()
        print(f"Received prediction result from backend: {prediction_result}")

        # Return the prediction service's response to the frontend
        return jsonify(prediction_result), response.status_code

    except requests.exceptions.RequestException as e:
        print(f"Error communicating with prediction service: {e}")
        return jsonify({"error": "Failed to connect to prediction service", "details": str(e)}), 503
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return jsonify({"error": "An internal server error occurred", "details": str(e)}), 500



if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=3000)
