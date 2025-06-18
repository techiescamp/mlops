# MLOPs Project

## Folder Structure:

```bash
mlops_project/
├── assets/
├── feature_store/
│   ├── data/
│   ├── __init__.py
│   ├── feature_order.py
│   ├── feature_store.yaml
│   ├── features.py
├── frontend/
├── kserve/
│   ├── predictor.py
├── mlruns/
├── monitoring/
│   ├── __init__.py
│   ├── inference_logs.csv
│   ├── logger.py
├── prediction-service/
│   ├── __init__.py
│   ├── app.py
├── raw_data/
├── src/
│   ├── __init__.py
│   ├── data_analysis.py
│   ├── data_preperation.py
│   ├── data_validation.py
│   ├── pipeline.py
│   ├── train_model.py
├── venv/
├── .gitignore
├── output.txt
├── Readme.md
├── requirements.txt


```

## Steps to run the code:

### Create `venv`
```bash
python -m venv venv

source venv/Scripts/activate   # (bash)
```

### Install Depeendencies
```bash
pip install -r requirements.txt
```

List of dependencies
```
pandas
feast
numpy
scikit-learn
fastparquet # pandas to support parquet
mlflow
```

### 1. `src/data_engg_pipeline.py`

```bash
# at root directory 'mlops_project/'

python -m src.data_engg_pipeline
```

This will create raw_data -> employee_attrition.csv (combined train.csv and test.csv)

### 2. Feature-Store

On new terminal, go to project directory `feature-store/`

```bash
cd feature-store

feast apply
feast materialize-incremental 2025-12-31T23:59:59
```

Then again run server

```bash
python main.py
```

For default server run without main.py

```bash
feast serve
```

This will run the server at default port `http://localhost:6566`


### 3.a. (Optional: For local MLflow setup) using `http://localhost:5000`

This module runs when mlflow server is already up and running or else create new one by opening new terminal and run this mlflow command

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
```

### 3.b. MLflow with cluster IP:

Since it is authenticated with Azure, have to login to azure-cli 

```bash
pip install azure-identity
pip install azure-storage-blob
```

- Run the `$env` to authenticate azure.
- Then run the following pipeline, so that our model will be registered.


### 4. `src/pipeline.py`
Note: In same mlops_project/ folder (root directory run this code)


On new terminal run the command at root folder `mlops_project/`
```bash
cd ../  # should be mlops_project/ folder

python -m src.model_pipeline
```


### For IP address MLflow setup no need to run mlflow server.

### Fourth run `kserve/predictor.py`

```bash
# set .env variable as mlflow-artifact-url

cd kserve  # got to mlops_project/src/ directory
python predictor.py
```

### Fifth run `prediction-service/app.py` (backend) - flask

```bash
# set .env variable as KSERVE_URL

cd prediction_service  # go to mlops_project/  root directory and run command there

python app.py
```

### Frontend `frontend/app.py` (frotnend) - flask

```bash
# set .env variable as KSERVE_URL

cd frontend  
python app.py
```

# ------------------------------------------------------

Example for local MLflow development url - # model_uri="../mlruns/0/<run-id>/artifacts/attrition_model_pipeline",


### Services Are:

1. MLflow for Model Registry 
2. KServe 
3. Feast 
4. Postgres for offline store
5. Redis DB for Online Store
6. Prediction-Backend 
7. Frontend 


