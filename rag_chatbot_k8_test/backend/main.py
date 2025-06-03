import os
import requests
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_openai import AzureChatOpenAI, AzureOpenAIEmbeddings
from components.document_loader import DocumentLoader
from components.text_splitter import TextSplitter
from components.vector_store import VectorStoreManager
from components.rag_components import RAGComponents
from components.utils import estimate_cost, clean_markdown
from components.runtime_evaluator import RuntimeEvaluator

# load env variables
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_CHAT_DEPLOYMENT = os.getenv("AZURE_CHAT_DEPLOYMENT")
AZURE_EMBEDDING_DEPLOYMENT = os.getenv("AZURE_EMBEDDING_DEPLOYMENT")
AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION")
AZURE_EMBEDDING_VERSION = os.getenv("AZURE_EMBEDDING_VERSION")

INPUT_COST = 0.00015 / 1000
OUTPUT_COST = 0.0006 / 1000

# Azure AI configuration
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_CHAT_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    openai_api_version=AZURE_OPENAI_API_VERSION    
)

embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=AZURE_ENDPOINT,
    deployment=AZURE_EMBEDDING_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    openai_api_version=AZURE_EMBEDDING_VERSION
)

# initialize components
document_loader = DocumentLoader(
    docs_folder="k8_docs/en",
    filename_embeddings_path="filename_embeddings.pkl",
    embedding_model=embedding_model
)

text_splitter = TextSplitter(
    chunk_size=1000,
    chunk_overlap=50
)

vector_store_manager = VectorStoreManager(
    vector_store_path='./faiss-db',
    embedding_model=embedding_model,
    batch_size=100
)
# testing
runtime_evaluator = RuntimeEvaluator(embedding_model=embedding_model)

# fastapi app
app = FastAPI()

# enable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class QueryRequest(BaseModel):
    query: str
    isEvaluate: bool = True # whether to compute runtime evaluation metrics

@app.post("/query")
async def query_rag(request: QueryRequest):
    print('🧑🏻‍🦱 query: ', request.query)
    try:
        query = request.query
        # testing
        # is_need_evaluation = request.isEvaluate
        
        # vector_store = vector_store_manager.load_vector_store(documents) 
        vector_store = requests.post(f"http://localhost:8001/search", json={"query": query})
        print("vector-db: ", vector_store)

        # rag components
        retriever = RAGComponents.retrieve_documents(vector_store, k=15)
        memory = RAGComponents.create_memory()
        augmentation_chain = RAGComponents.create_augmentation_chain(retriever, llm, memory)

        # retrieve documnets and create context
        search_docs = retriever.invoke(query)
        context = "\n\n".join(clean_markdown(doc.page_content) for doc in search_docs)

        # prepare input text for cost estimation
        input_text = f"Chat History: {memory.load_memory_variables({})['chat_history']}\nContext: {context}\nQuestion: {query}"

        # invoke augmentation chain
        result = augmentation_chain.invoke(query)

        # estimate cost
        input_tokens, output_tokens, cost = estimate_cost(input_text, result, AZURE_CHAT_DEPLOYMENT, input_cost=INPUT_COST, output_cost=OUTPUT_COST)

        # Save conversation context
        memory.save_context({"question": query}, {"answer": result})

        # testing
        evaluation = {}
        # retrieved_docs = documetns retrieved from RAG component
        # selected_Docs = docs selected at starting of the pipeline before RAG component
        if is_need_evaluation:
            evaluation = runtime_evaluator.evaluate(query=query, retrieved_docs=search_docs, selected_docs=documents, answer=result, context=context)
            print("Performance Evaluation: \n", evaluation)

        return {
            "answer": result,
            "sources": [doc.metadata['source'] for doc in search_docs],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": cost,
            "evaluation": evaluation if is_need_evaluation else None,
        }
    except Exception as e:
        print(f"Error processing query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health_check():
    return {"status": "healthy"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
