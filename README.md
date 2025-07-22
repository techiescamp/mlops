# MLOPS Projects

<div align="center">
  <img src="https://img.shields.io/github/stars/techiescamp/mlops.svg?style=for-the-badge" alt="GitHub Stars" />
  <img src="https://img.shields.io/github/forks/techiescamp/mlops.svg?style=for-the-badge" alt="GitHub Forks" />
  <img src="https://img.shields.io/github/contributors/techiescamp/mlops.svg?style=for-the-badge" alt="Contributors" />
  <img src="https://img.shields.io/github/last-commit/techiescamp/mlops.svg?style=for-the-badge" alt="Last Commit" />
  <img src="https://img.shields.io/badge/python-3.x-blue?style=for-the-badge" alt="Python Version" />
</div>

## Overview

This repository provides a comprehensive approach to Machine Learning Operations (MLOps), integrating machine learning models into production with automation, monitoring, and scalability. It covers best practices, CI/CD pipelines, model versioning, and deployment strategies.


## Table of Contents
- [Introduction](#introduction)
- [Projects In this Repo](#projects)
- [Installation](#installation)
- [Future Enhancements](#future-enhancements)
- [Contribution](#contribution)
- [License](#license)

## Introduction

This repository includes multiple MLOps projects, each focusing on different aspects of machine learning model development, deployment, and monitoring. The projects are structured as follows:

## Projects In this Repo

### **1. Employee Attrition Prediction**
    - Uses Logistic Regression for predicting employee attrition.
    - Implements Flask for web-based model interaction.
    - Features automated data preprocessing, model training, and deployment using Docker and Kubernetes.

### **2. LLM-Based Simple models using Hugging Face**
    - Built simple LLM project using Hugging Face's open source models on
        - text summarization, 
        - text generation, 
        - sentiment-analysis, 
        - question-answering and 
        - table question-answering models

    - Deployed via `React` (frontend) and `Node.js,Express.js` (backend) for seamless user experience.


### **3. Fine-Tuning using Azure OpenAI model**
Fine-tuning is the process of training a pre-trained model with our own custom dataset so that model gives context-aware response and avoid hallucinations.        

    - Preparing and uploading dataset into Azure
    - Training the model
    - Tracking events and checkpoints of model
    - Continue fine-tuning with the previously trained model
    - Deploy and test the model


### **4. RAG project**
The Retrieval-Augmented Generation (RAG) workflow enhances efficiency by dynamically fetching relevant data from a knowledge base and integrating it with a language model to produce precise, context-aware responses, eliminating the need for extensive retraining.

    - Developed DocuMancer AI, utilizing the RAG workflow to retrieve content from GitHub .md files.
    - Optimized DocuMancer AI by streamlining the RAG workflow for seamless processing of Kubernetes documentation.


## Installation

**Prerequisites**

    - Python +3.10
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

    Go to the project directory of your choice and start working on it.


## Future Enhancements
- Automated ML Pipelines using DVC & MLflow.
- Continuous Integration & Deployment (CI/CD) with GitHub Actions.
- Model Versioning and tracking experiments.
- Cloud Deployment with Docker & Kubernetes.
- Monitoring & Logging with Prometheus & Grafana.


## Contribution
We welcome contributions from the security community. Please read our [Contributing Guidelines](./CONTRIBUTION.md) before submitting pull requests.


## License

This project is open-source and available under the [MIT License](./LICENCE).
[&copy;2025 www.techiescamp.com/](www.techiescamp.com/)
