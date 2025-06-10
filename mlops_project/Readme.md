# MLOPs Project

## Folder Structure:

```bash
employee_attrition_mlops/
└── raw_data
    └── train.csv
    └── test.csv
├── feature_store/
    └── feature_store.yaml
    └── features.py
    ├── data/
        └── employee_preprocessed_data.parquet
        └── online_store.db
        └── registry.db
└── src
    └── data_preparation.py
    └── train_model.py
├── frontend/
    └── app.py
    └── templates/
        └── index.html
|_ .env
|_ readme.md

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

### First run `src/data_preparation.py`
```bash
cd src

python data_preperation.py
```

### Second run Feature-Store

Go to project directory `feature-store/`

```bash
cd feature-store

feast apply
feast materialize 2024-01-01T00:00:00 2025-06-05T23:59:59
```

### Third run `src/train_model.py`

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 127.0.0.1 --port 5000


cd src  # got to mlops_project/src/ directory
python train_model.py
```

### Fourth run `kserve/predictor.py`

```bash
# set .env variable as mlflow-artifact-url

cd kserve  # got to mlops_project/src/ directory
python predictor.py
```

### Fifth run `prediction-service/app.py` (backend) - flask

```bash
# set .env variable as KSERVE_URL

cd prediction-service  # got to mlops_project/src/ directory
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
