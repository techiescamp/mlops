from feast.feature_server import start_server 
from feast.feature_store import FeatureStore
from dotenv import load_dotenv
from datetime import datetime
import os

# Import all definitions from your features module
from features import employee, employee_features_fv, employee_attrition_fs

load_dotenv()

# FEAST_SERVER_HOST = os.environ.get("FEAST_SERVER_HOST", "localhost")
# FEAST_SERVER_PORT= int(os.environ.get("FEAST_SERVER_PORT", "5050"))

FEAST_SERVER_HOST = "localhost"
FEAST_SERVER_PORT= 5050


# FEAST_SERVER_HOST = "localhost"
# FEAST_SERVER_PORT= 5050


def main():
    store = FeatureStore(repo_path=".")
    # Apply entities, feature views, and feature services
    store.apply([
        employee,
        employee_features_fv,
        employee_attrition_fs,
    ])
    store.materialize_incremental(end_date=datetime.now())
    start_server(
        store=store, 
        host=FEAST_SERVER_HOST, 
        port=FEAST_SERVER_PORT,
        no_access_log=False,
        workers=1,
        keep_alive_timeout=60,
        registry_ttl_sec=10,
        tls_key_path=None,
        tls_cert_path=None,
        metrics=False
    )    

if __name__ =="__main__":
    main()
    