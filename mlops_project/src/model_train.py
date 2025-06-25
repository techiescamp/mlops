# train_model.py
import time
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
#  for monitoring
import psutil
import pynvml


def model_training(X_train, y_train, classifier):
    model_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('classifier', classifier)
    ])

    # start cpu/gpu monitoring
    cpu_start = psutil.cpu_percent(interval=None)
    mem_start = psutil.virtual_memory().used / (1024 ** 2)  # in MB
    
    # Initialize NVML for GPU monitoring
    try:
        pynvml.nvmlInit()
        count = pynvml.nvmlDeviceGetCount()
        print("Device count:", count)
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        gpu_start = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_mem_start = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)  # MB
    except pynvml.NVMLError as e:
        print("GPU monitoring not found start of monitoring is skipped!")
        gpu_start = 0
        gpu_mem_start = 0

    model_pipeline.fit(X_train, y_train)

    try:
        gpu_end = pynvml.nvmlDeviceGetUtilizationRates(handle).gpu
        gpu_mem_end = pynvml.nvmlDeviceGetMemoryInfo(handle).used / (1024 ** 2)  # MB
        pynvml.nvmlShutdown()
    except pynvml.NVMLError as e:
        print("GPU end of monitoring skipped")
        gpu_end = 0
        gpu_mem_end = 0


    # cpu end monitoring
    cpu_end = psutil.cpu_percent(interval=None)
    mem_end = psutil.virtual_memory().used / (1024 ** 2)  # in MB

    # total cpu/gpu usages
    cpu_usage = (cpu_end + cpu_start) / 2 
    cpu_memory_usage = (mem_end + mem_start) / 2  
    gpu_usage = (gpu_end + gpu_start) / 2
    gpu_memory_usage = (gpu_mem_end + gpu_mem_start) / 2

    # Calculate uptime in seconds
    uptime_seconds = time.time() - psutil.boot_time()

    system_metrics = {
        "cpu_usage": cpu_usage,
        "gpu_usage": gpu_usage,
        "cpu_memory_usage_mb": cpu_memory_usage,
        "gpu_memory_usage_mb": gpu_memory_usage,
        "uptime_seconds": uptime_seconds
    }

    coefficients = None
    if hasattr(model_pipeline.named_steps['classifier'], 'coef_'):
        coefficients = model_pipeline.named_steps['classifier'].coef_[0]
    print('coefficients:', coefficients)

    return model_pipeline, coefficients, system_metrics
