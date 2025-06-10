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
│       └── employee_preprocessed_data.parquet
└── src
    └── data_preparation.py
    └── train_model.py
├── frontend/
    └── app.py
    └── templates/
        └── index.html

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
cd src  # got to mlops_project/src/ directory
python train_model.py
```

