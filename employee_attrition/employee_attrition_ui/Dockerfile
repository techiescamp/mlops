FROM python:3.12-slim
WORKDIR /app
COPY requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt
COPY static ./static
COPY templates ./templates
COPY app.py .
EXPOSE 5000
CMD ["python", "-m", "flask", "run", "--host=0.0.0.0", "--port=5000"]
