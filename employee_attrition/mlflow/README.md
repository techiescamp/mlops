# 🧠 Employee Attrition Prediction using Flask & MLflow

This project demonstrates how to deploy a machine learning model using **Flask** for the frontend and backend, and **MLflow** for model tracking and artifact management. The model predicts whether an employee is likely to **stay** or **leave** a company based on various HR-related features.

## 🚀 Features

- Frontend rendered using Flask templating (`index.html`)
- ML model served with Flask and integrated using MLflow
- Artifacts such as scaler, encoder, and feature list downloaded from MLflow
- Preprocessing logic mirrors the training phase for accurate inference

## 🛠️ Tech Stack

- Python
- Flask
- Pandas
- MLflow
- HTML/CSS (for frontend UI)

## 📁 Folder Structure

project/ ├── templates/ │ └── index.html # Frontend UI ├── app.py # Flask backend


## 🧪 How to Run

1. **Install dependencies:**

```bash
pip install flask flask-cors pandas mlflow
```

2. **Start MLflow Tracking Server (if not running)**:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

3. **Open new termial and run the mlflow-model**:

```bash
python train.py
```

**Note: First time run the model on MLflow UI.**

4. **Another terminal Run the Flask app**:

Go to `cd frontend/` and run below command

```bash
python app.py
```

or 

```bash
flask run
```


## 📊 Prediction Output
The prediction result will be:

✅ Stayed

❌ Left

Based on your input via the frontend form.

## 🧠 Credits
Model trained and tracked using MLflow. Flask used for quick and scalable deployment.

## 📄 License
This project is licensed under the MIT License.
