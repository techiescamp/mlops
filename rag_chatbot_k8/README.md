
## RAG powered chatbot by webscraping the Kubernetes documents

**RAG** - **Retrieval Augmented Generation** is a workflow that's simplifies the use of generating content by taking one's own knowledge base as reference. In fine-tuning, we have to store large content of data in the model, which can be resource-intensive and time-consuming, RAG streamlines the process by retrieving relevant information from a knowledge base dynamically and combining it with a language model to generate accurate, context-specific responses without the need for extensive retraining.


## Github Actions workflow steps:
-------------------------------------------
On push and pull requests follow the below actions:

Backend: 
- python version - 3.13.3
- Install dependencies - `pip install -r requirements.txt`
- Run backend code - `pyton app.py` or `uvicorn main:app --reload`

Frontend:
- node version - 22.14.0
- Install dependencies - `npm install`
- Run code - `npm run build`

Deploy:
- deploy both backend and frontend


## Project Workflow
-----------------------------------------------------------------------------------

![rag-workflow-gif](assets/rag-1.gif)


## Getting Started
-----------------------------------------------------------------------------
To set up and run the project, follow these steps:

**Prerequisites**
- Python 3.10+
- Node for frontend v 22.14.0
- Install dependencies: `pip install -r requirements.txt`
- Azure OpenAI API credentials (set in a .env file)
- Markdown files in the docs/ directory

## Setup files
### 1. Clone the repository

```bash
git clone https://github.com/techiescamp/mlops.git
cd rag_chatbot_k8
```

### 2. Install dependencies
### i. sync-docs folder:

- Create .env

```bash
# .env
AZURE_ENDPOINT=<http://your-azure-resource>
AZURE_API_KEY=<your-api-key-from-azure>

AZURE_EMBEDDING_DEPLOYMENT=<text-embedding-model-model>
AZURE_EMBEDDING_VERSION=<text-embedding-model-version-date-on-azure>

K8_URL=<k8-github-repo-url>
VECTOR_DB_URL=<vector-db-url>
```

- Create venv 

```bash
python -m venv venv

venv/Scripts/activate (for windows)
source ./venv/Scripts/activate (for bash)
```

- Install requirements.txt

```bash
pip install -r requirements.txt
```

- Run code

```bash
python index.py
```

<!-- ------------- -->

### vector-db folder

- Create .env

```bash
# .env
AZURE_ENDPOINT=<http://your-azure-resource>
AZURE_API_KEY=<your-api-key-from-azure>

AZURE_EMBEDDING_DEPLOYMENT=<text-embedding-model-model>
AZURE_EMBEDDING_VERSION=<text-embedding-model-version-date-on-azure>

PORT=<port>
HOST=<vector-db-url-name>
```

Create venv 

```bash
python -m venv venv

venv/Scripts/activate (for windows)
source ./venv/Scripts/activate (for bash)

```

- Install requirements.txt

```bash
pip install -r requirements.txt
```

Run code

```bash
python index.py

or 

uv run index.py (preferred)
```

<!-- ------------ -->

### Main Backend folder:

- Create .env

```bash
# .env
AZURE_ENDPOINT=<http://your-azure-resource>
AZURE_API_KEY=<your-api-key-from-azure>

AZURE_CHAT_DEPLOYMENT=<gpt-model>
OPENAI_API_VERSION=<gpt-model-version-date-on-azure>

VECTOR_DB_URL=<backend-url>
HOST=<dns-name>
PORT=<port>
```

- Create venv 

```bash
python -m venv venv

venv/Scripts/activate (for windows)
source ./venv/Scripts/activate (for bash)
```

- Install requirements.txt

```bash
pip install -r requirements.txt
```


- Run code

```bash
python main.py

or 

uv run main.py (preferred)
```

<!-- ----------------------- -->

### For Frontend Setup

- If you are in different folder => `rag_Chatbot/frontend/` 
- If you are in same folder => `frontend/`

```bash
cd frontend
```

- Install dependencies
```bash
npm install
```

- Run the frontend application
```bash
npm start (for developement environement)

or

npm run build (for production environment)
```

```

### Example Query
```json
{
  "query": "What is memoization?"
}
```

### Example Response

```json
{
  "answer": "Memoization is an optimization technique used primarily to speed up computer programs by storing the results of expensive function calls and reusing them when the same inputs occur again.",
  "sources": [
    {"filename": "concepts.md", "chunk_id": 1},
    {"filename": "optimization.md", "chunk_id": 3}
  ]
}
```


### Why Use This Chatbot?
----------------------------------
- **Accurate Answers:** Retrieves and generates responses based on your project’s documentation.
- **Context-Aware:** Maintains conversation history for coherent interactions.
- **Efficient:** Uses vector search (FAISS) for fast document retrieval.
- **Cost-Conscious:** Tracks token usage and estimates costs for Azure OpenAI API calls.
- **Customizable:** Easily adapt the pipeline to other document formats or LLMs.
=======
### For sync-docs:

1. create .env

2. create venv 

```bash
python -m venv venv

venv/Scripts/activate (for windows)
source ./venv/Scripts/activate (for bash)
```

3. run code

```bash
python index.py
```

<!-- ------------- -->

### For vector-db:

1. create .env

2. create venv 

```bash
python -m venv venv

venv/Scripts/activate (for windows)
source ./venv/Scripts/activate (for bash)

```

3. run code

```bash
python index.py

or 

uv run index.py (preferred)
```

<!-- ------------ -->

### For backend:

1. create .env

2. create venv 

```bash
python -m venv venv

venv/Scripts/activate (for windows)
source ./venv/Scripts/activate (for bash)

```

3. run code

```bash
python main.py

or 

uv run main.py (preferred)
```


### Contribution
-----
We welcome contributions from the security community. Please read our ![Contributing Guidelines](../CONTRIBUTION.md) before submitting pull requests.

### License
This project is licensed under the ![MIT License](../LICENCE). See the  file for details
=======

