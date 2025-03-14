FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY model_class.py .
COPY server.py .
COPY models /mnt/models
CMD ["python", "server.py"]