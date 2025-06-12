# monitoring/utils.py

import time
import csv
from datetime import datetime
import os

MONITOR_LOG_FILE = os.path.join(os.path.dirname(__file__), 'inference_logs.csv')

def log_metrics(data: dict):
    log_exists = os.path.isfile(MONITOR_LOG_FILE)

    with open(MONITOR_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(['timestamp', 'latency_ms', 'status', 'prediction'])  # header
        writer.writerow([
            datetime.utcnow().isoformat(),
            data.get('latency_ms'),
            data.get('status'),
            data.get('prediction'),
        ])
