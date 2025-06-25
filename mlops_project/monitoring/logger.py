# monitoring_service.py
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from pydantic import BaseModel
from typing import Optional, Any
import os
from datetime import datetime, timezone
import csv
import json

app = FastAPI()

# Configuration for log file
MONITOR_LOG_DIR = "logs" # Or an absolute path like '/var/log/monitoring'
MONITOR_LOG_FILE = os.path.join(MONITOR_LOG_DIR, 'inference_logs.csv')
DRIFT_LOG_FILE = os.path.join(MONITOR_LOG_DIR, "data_drift.json")

# Ensure the log directory exists
os.makedirs(MONITOR_LOG_DIR, exist_ok=True)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# Pydantic model for incoming log data
class LogData(BaseModel):
    latency_ms: float
    status: str
    prediction: Optional[Any] = None # Use Any as prediction can be int, float, etc.


def _write_log(data: LogData):
    log_exists = os.path.isfile(MONITOR_LOG_FILE)

    with open(MONITOR_LOG_FILE, mode='a', newline='') as file:
        writer = csv.writer(file)
        if not log_exists:
            writer.writerow(['timestamp', 'latency_ms', 'status', 'prediction'])  # header
        writer.writerow([
            datetime.now(datetime.timezone.utc).isoformat(),
            data.latency_ms,
            data.status,
            data.prediction,
        ])


@app.post("/log")
async def log_inference(log_data: LogData):
    try:
        _write_log(log_data)
        print(f"Log received: {log_data.dict()}")
        return {"message": "Log recorded successfully"}
    except Exception as e:
        print(f"Error writing log: {e}")
        return {"error": "Failed to record log"}, 500


@app.get("/metrics")
async def get_metrics():
    metrics = {}
    if not os.path.exists(MONITOR_LOG_FILE) and not os.path.exists(DRIFT_LOG_FILE):
        return {"error": "Metrics file not found. No logs yet."}, 404
    
    # Load inference logs if available
    if os.path.exists(MONITOR_LOG_FILE):
        with open(MONITOR_LOG_FILE, newline='') as file:
            reader = csv.DictReader(file)
            metrics["inference_logs"] = list(reader)
    else:
        metrics["inference_logs"] = []

    # Load drift alerts if available
    if os.path.exists(DRIFT_LOG_FILE):
        with open(DRIFT_LOG_FILE, 'r') as file:
            drift_entries = [json.loads(line) for line in file if line.strip()]
        metrics["drift_alerts"] = drift_entries
    else:
        metrics["drift_alerts"] = []

    return {"metrics": metrics}


@app.post("/alert")
def drift_alert(data: dict):
    if(data): 
        os.makedirs(MONITOR_LOG_DIR, exist_ok=True)
        print("ALERT: Recieved data with drift detected")
        log_entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "type": "drift_alert",
            "data": data
        }
        try:
            with open(DRIFT_LOG_FILE, "a") as f:
                f.write(json.dumps(log_entry) + "\n")
            return {"message": "Drift alert logged successfully"}
        except Exception as e:
            print(f"Error logging drift alert: {e}")
            return {"Error": "Failed logging drift alert"}, 500
    else:
        print("ALERT: No drift detected")
        return {"message": "No drift detected"}



if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001) # Runs on a different port than prediction backend