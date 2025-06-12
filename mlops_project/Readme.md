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

### first run `data_preparation.py`

```bash
cd src

python data_preparation.py
```

### Second run Feature-Store

On new terminal, go to project directory `feature-store/`

```bash
cd feature-store

feast apply
feast materialize 2024-01-01T00:00:00 2025-06-05T23:59:59
```

### Third run `src/pipeline.py`
Note: In same mlops_project/ folder (root directory run this code)

```bash
cd ../  # should be mlops_project/ folder

python -m src.pipeline
```

### (Optional: For local MLflow setup) using `http://localhost:5000`

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000
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

cd ../  # go to mlops_project/  root directory and run command there

python -m prediciton-service.app
```

### Frontend `frontend/app.py` (frotnend) - flask

```bash
# set .env variable as KSERVE_URL

cd frontend  
python app.py
```

# ------------------------------------------------------

Example for local MLflow development url - # model_uri="../mlruns/0/<run-id>/artifacts/attrition_model_pipeline",
