import os
import pickle
import sys
import numpy as np
import pandas as pd
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# sart flask app
app = Flask(__name__)

# Cors
CORS(app)

# get the absolute path of the current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# move one level up to the project root directory
PROJECT_DIR = os.path.dirname(BASE_DIR)

# Define the path to the model file inside the "model" directory
MODEL_DIR = os.path.join(PROJECT_DIR, "employee_attrition_kserve_model")
sys.path.append(MODEL_DIR)

# import model class here
from model_class import EmployeeAttritionModel
model_path = os.path.join(MODEL_DIR, "my_model_lr.pkl")

# load the model and the scaler
if os.path.isfile(model_path):
    with open(model_path, "rb") as f:
        model = pickle.load(f)
else:
    raise FileNotFoundError(f"Model file not found at {model_path}")


# home page
@app.route('/')
def index():
    return render_template('index.html')

# start route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json(force=True)
    predicition = model.predict([data])
    result = "Left" if predicition[0] else "Stayed"
    return jsonify({"prediction": result})

if __name__ == "__main__":
    app.run(debug=True)