## RAG powered chatbot using kubernetes docs

Note - The kubernetes docs can be found ![here](https://github.com/kubernetes/website/tree/main/content/en/docs). Copy the 'en' version of documents and save it in your project folder to use this DocuMancer AI locally.


**RAG** - **Retrieval Augmented Generation** is a workflow that's simplifies the use of generating content by taking one's own knowledge base as reference. In fine-tuning, we have to store large content of data in the model, which can be resource-intensive and time-consuming, RAG streamlines the process by retrieving relevant information from a knowledge base dynamically and combining it with a language model to generate accurate, context-specific responses without the need for extensive retraining.

## Purpose of DocuMancerK8s AI
The DocuMancerK8s AI project aims to enhance access to Kubernetes documentation by utilizing a Retrieval-Augmented Generation (RAG) workflow. This approach enables efficient content retrieval from Kubernetes markdown files, streamlining the process of navigating and extracting information from the official documentation. 

The project focuses on improving user interaction with Kubernetes resources, making it easier to understand and apply complex concepts. The Kubernetes documentation, which serves as the primary data source, is available for download at https://github.com/kubernetes/website/tree/main/content/en/docs. 

This initiative seeks to empower users by providing a more intuitive and effective way to engage with Kubernetes documentation, supporting both learning and practical implementation.

## Why Use This Chatbot?
----------------------------------
- **Accurate Answers:** Retrieves and generates responses based on your project’s documentation.
- **Context-Aware:** Maintains conversation history for coherent interactions.
- **Efficient:** Uses vector search (FAISS) for fast document retrieval.
- **Cost-Conscious:** Tracks token usage and estimates costs for Azure OpenAI API calls.
- **Customizable:** Easily adapt the pipeline to other document formats or LLMs.


## Github Actions workflow steps:
-------------------------------------------
When forking the project follow the below actions:

Backend: 
- python version - 3.13.3
- Install dependencies - `pip install -r requirements.txt`
- Run backend code - `python app.py` or `uvicorn main:app --reload`

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

## Installation
1. Clone the repository

```bash
git clone https://github.com/your-username/rag-chatbot.git
cd rag-chatbot
```

2. Install dependencies

```bash
pip install -r requirements.txt
```

3. Set up environment variables in a .env file:

```plaintext
AZURE_ENDPOINT=<azure-endpoint>
AZURE_API_KEY=<azure-openai-api-key>
AZURE_DEPLOYMENT=<model-name:(e.g: gpt-4o-mini)>
OPENAI_API_VERSION=<version-date-in-azure:(e.g: 2024-13-3)>
GITHUB_REPOSITORY=<github-repo>
```

4. Place your Markdown documentation in the `k8_docs/en/` directory

## Running the Application

1. Start the FastAPI server

```
python app.py
```

or

```
uvicorn main:app --reload
```

2. Access the API at http://localhost:8000.
3. Use a frontend (e.g., a React app) or tools like Postman to send queries to POST /query

## Example Query
```json
{
  "query": "Explain service traffic policy in k8"
}
```

## Example Response
  The FastAPI server sends the JSON response to the client.

```json
{
  "answer": "Service Internal Traffic Policy in Kubernetes is a feature that allows for efficient communication between Pods on the same node within a cluster. When two Pods need to connect, using this policy...",
  "sources": [
    {"filename": "service-pods-2.md", "chunk_id": 2},
    {"filename": "service-pods-3.md", "chunk_id": 3},
    {"filename": "service-pods-11.md", "chunk_id": 11}
  ],
  "input_tokens": 123,
  "output_tokens": 78,
  "estimated_cost": 0.0048
}

```

## Frontend Setup

- If you are in different folder => `rag_chatbot_k8/frontend/` 
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


## Detailed Explanation of Workflow
----------------------------------------------------------------------------
### Client (Frontend)
The process starts with a client (e.g., a React app at http://localhost:3000) sending requests to the FastAPI backend.

### FastAPI Application
The entry point is the FastAPI app, which handles CORS middleware and main endpoint:
`POST /query:` The core endpoint for processing user queries with the RAG pipeline.

### Query Processing (POST /query)
The client sends a JSON payload (e.g., {"query": "What is memoization?"}) to /query.
The request is validated using QueryRequest (Pydantic model).

### RAG Pipeline
The RAG pipeline is modularized, with code organized in the components/ folder.

`Load Markdown Docs`: In `dcoument_loader.py` file, the load_md_files() reads .md files from the `k8_docs/en` directory storing them as a list of dictionaries containing filenames and content.

`Select Relevant Documents`: 
The document_loader.py module handles document selection:
  - If filename_embeddings.pkl is not cached locally, precompute() generates filename embeddings for faster future access.
  - select_relevant_files() picks the top 5 documents (k=5) based on the user’s query and passes them to text_splitter().

`Text Splitting`: text_splitter() uses RecursiveCharacterTextSplitter to break the Markdown content into smaller chunks (500 characters, 100 overlap), creating Document objects with metadata.

`Vector Store (FAISS)`: load_vector_store() either loads an existing FAISS index from ./faiss-db or creates a new one using embeddings from HuggingFaceEmbeddings (all-MiniLM-L6-v2 model).

`Retriever`: retrieve_documents() configures the FAISS vector store as a retriever to fetch the top 3 relevant document chunks based on the query.

`Memory`: create_memory() initializes ConversationBufferMemory to store the last 5 question-answer pairs, maintaining chat history.

`Augmentation Chain`: augmentation() builds a chain with:
    - A ChatPromptTemplate combining chat history, context (retrieved docs), and the query.
    - Then set `chaining` feature of langchain to combine the events of fetching files to calling LLM model:
        - The retriever to fetch context.
        - The AzureChatOpenAI LLM to generate answers.
        - A StrOutputParser to format the output.
        - AzureChatOpenAI LLM: The LLM (configured with Azure credentials from .env) processes the prompt and generates a response based on the context and history.

### Cost Estimation
`estimate_cost()`: This function uses tiktoken to count input and output tokens, calculating the cost based on predefined rates (INPUT_TOKEN_COST and OUTPUT_TOKEN_COST).

### Response Generation
The response includes:
    `answer`: The LLM-generated response.
    `sources`: Metadata of retrieved documents (e.g., filenames and chunk IDs).
    `input_tokens, output_tokens, estimated_cost`: Token usage and cost details.
The query and answer are saved to memory for future context.


## RAG Performance Metrics
-------------------------------------------------------------------------
### A. Retrieval Metrics: 
Measures how well the retriever fetches relevant documents.

1. Recall@k: Proportion of relevant documents retrieved in the top-k.

2. Precision@k: Proportion of top-k documents that are relevant.

3. MRR (Mean Reciprocal Rank): Average of reciprocal ranks of the first relevant result.

### B. Generation Metrics:
Measures how well the LLM generate response using relvant docuemnts.

1. **Semantic Similarity:** Cosine similarity between the generated answer and the retrieved context, assessing grounding.

2. **Out-of-Context:** Checks for the 🌟 emoji to detect retriever failures.


### Implementation steps of metrics:
Metrics ae implemented in `runtime_evaluator.py` and `utils.py`. 

**`runtime_evaluator.py` file:**

- `_get_relevant_sources:` Uses a list comprehension to filter documents with similarity above similarity_threshold.
- `_get_query_doc_similarity:` Computes average similarity with a single loop.
- `_get_document_diversity:` Simplifies pairwise similarity calculation using a list comprehension.
- `evaluate_retrieval` and `evaluate_generation:` Combine metrics clearly without redundant computations.

**`utils.py` file:** 
- All similarity calculations use compute_semantic_similarity and compute_text_overlap.


### Contributing
-----------------------------------------------------
We welcome contributions from the security community. Please read our ![Contributing Guidelines](../CONTRIBUTION.md) before submitting pull requests.

### License
----------------------------------------------------
This project is licensed under the ![MIT License](../LICENCE). See the  file for details


langchain_core
langchain_community
langchain