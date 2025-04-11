## RAG powered chatbot for project documents

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

![rag-workflow-gif](/assets/rag-1.gif)


## Getting Started
-----------------------------------------------------------------------------
To set up and run the project, follow these steps:

**Prerequisites**
- Python 3.10+
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
AZURE_DEPLOYMENT=<model-name:(e.g: gpt-4)>
OPENAI_API_VERSION=<version-date-in-azure:(e.g: 2024-13-3)>
GITHUB_REPOSITORY=<github-repo>
```

4. Place your Markdown documentation in the `docs/` directory

### Running the Application

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
  ],
  "input_tokens": 150,
  "output_tokens": 50,
  "estimated_cost": 0.0025
}
```


### Detailed Explanation of Workflow
----------------------------------------------------------------------------
### Client (Frontend):
    The process starts with a client (e.g., a React app at http://localhost:3000) sending requests to the FastAPI backend.

### FastAPI Application:
    The entry point is the FastAPI app, which handles CORS middleware and main endpoint:
        - `POST /query:` The core endpoint for processing user queries with the RAG pipeline.

### Query Processing (POST /query):
    - The client sends a JSON payload (e.g., {"query": "What is memoization?") to /query.
    - The request is validated using QueryRequest (Pydantic model).

### RAG Pipeline:
    - `Load Markdown Docs`: load_md_files() reads .md files from the docs/ directory into memory as a list of dictionaries (filename and content).
    - `Text Splitting`: text_splitter() uses RecursiveCharacterTextSplitter to break the Markdown content into smaller chunks (500 characters, 100 overlap), creating Document objects with metadata.
    - `Vector Store (FAISS)`: load_vector_store() either loads an existing FAISS index from ./faiss-db or creates a new one using embeddings from HuggingFaceEmbeddings (all-MiniLM-L6-v2 model).
    - `Retriever`: retrieve_documents() configures the FAISS vector store as a retriever to fetch the top 3 relevant document chunks based on the query.
    - `Memory`: create_memory() initializes ConversationBufferMemory to store the last 5 question-answer pairs, maintaining chat history.
    - `Augmentation Chain`: augmentation() builds a chain with:
        - A ChatPromptTemplate combining chat history, context (retrieved docs), and the query.
        - Then sed `chaining` feature of langchain to combine the events of fetching files to calling LLM model:
            - The retriever to fetch context.
            - The AzureChatOpenAI LLM to generate answers.
            - A StrOutputParser to format the output.
            - AzureChatOpenAI LLM: The LLM (configured with Azure credentials from .env) processes the prompt and generates a response based on the context and history.

### Cost Estimation:
`estimate_cost()`: This function uses tiktoken to count input and output tokens, calculating the cost based on predefined rates (INPUT_TOKEN_COST and OUTPUT_TOKEN_COST).

### Response Generation:
    The response includes:
        `answer`: The LLM-generated response.
        `sources`: Metadata of retrieved documents (e.g., filenames and chunk IDs).
        `input_tokens, output_tokens, estimated_cost`: Token usage and cost details.
    The query and answer are saved to memory for future context.


### Why Use This Chatbot?
----------------------------------
- **Accurate Answers:** Retrieves and generates responses based on your project’s documentation.
- **Context-Aware:** Maintains conversation history for coherent interactions.
- **Efficient:** Uses vector search (FAISS) for fast document retrieval.
- **Cost-Conscious:** Tracks token usage and estimates costs for Azure OpenAI API calls.
- **Customizable:** Easily adapt the pipeline to other document formats or LLMs.

### Return Response to Client:
    The FastAPI app sends the JSON response back to the client.

### Contributing
We welcome contributions! To get started:

1. Fork the repository.
2. Create a feature branch (git checkout -b feature/your-feature).
3. Commit your changes (git commit -m "Add your feature").
4. Push to the branch (git push origin feature/your-feature).
5. Open a pull request.

### License
This project is licensed under the MIT License. See the  file for details
