# MLOPS Projects
----------------------------------------

## OVerview

This repository provides a comprehensive approach to Machine Learning Operations (MLOps), integrating machine learning models into production with automation, monitoring, and scalability. It covers best practices, CI/CD pipelines, model versioning, and deployment strategies.

## Projects

This repository includes multiple MLOps projects, each focusing on different aspects of machine learning model development, deployment, and monitoring. The projects are structured as follows:

### **Employee Attrition Prediction**

        - Uses Logistic Regression for predicting employee attrition.
        - Implements Flask for web-based model interaction.
        - Features automated data preprocessing, model training, and deployment using Docker and Kubernetes.

### **LLM-Based Simple models using Hugging Face**

        - Built simple LLM project using Hugging Face's open source models on
            - text summarization, 
            - text generation, 
            - sentiment-analysis, 
            - question-answering and 
            - table question-answering models

        - Deploys via `React` (frontend) and `Node.js,Express.js` (backend) for seamless user experience.

## Installation & Setup

**Prerequisites**

    - Python 3.x
    - Docker & Kubernetes (Optional for Deployment)

**Steps**

    1. Clone the repository
    ```
    git clone https://github.com/techiescamp/mlops.git
    cd mlops
    ```

    2.  a virtual environment (Recommended)
    ```
    python -m venv venv
    source venv/bin/activate  # For macOS/Linux
    venv\Scripts\activate     # For Windows
    ```

    3. Install dependencies (if requirements.txt exists)
    ```
    pip install -r requirements.txt
    ```

    Go to directory on which project you needed and start working on it.

## Future Enhancement

    - Automated ML Pipelines using DVC & MLflow.
    - Continuous Integration & Deployment (CI/CD) with GitHub Actions.
    - Model Versioning and tracking experiments.
    - Cloud Deployment with Docker & Kubernetes.
    - Monitoring & Logging with Prometheus & Grafana.


## License

This project is open-source and available under the MIT License.
**&copy; www.techiescamp.com/**

