import pickle
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import numpy as np
import os
from typing import List, Any
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain.docstore import InMemoryDocstore
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


app = FastAPI()

# enable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Vector DB path
VECTOR_DB_PATH = "vector_store/index.faiss"
DOCS_STORE_PATH = "vector_store/docs.pkl"

# Azure embedding model
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_key=os.getenv("AZURE_API_KEY"),
    openai_api_version=os.getenv("AZURE_EMBEDDING_VERSION")
)

class EmbeddingItem(BaseModel):
    embedding: List[float]
    metadata: dict
    content: str

class QueryRequest(BaseModel):
    query: str
    top_k: int = 5


def load_vector_store():
    if os.path.exists(VECTOR_DB_PATH) and os.path.exists(DOCS_STORE_PATH):
        with open(DOCS_STORE_PATH, "rb") as f:
            docs = pickle.load(f)
        index = faiss.read_index(VECTOR_DB_PATH)
        return FAISS(embedding_model, index, InMemoryDocstore(docs), {})
    else:
        index = faiss.IndexFlatL2(1536) # embedding_model.embedding_dimension
        return FAISS(embedding_model, index, InMemoryDocstore({}), {})
   

vector_store = load_vector_store()


@app.post("/store")
async def store_embeddings(data: List[EmbeddingItem]):
    try:
        docs = [
            Document(
                page_content = item.page_content,
                metadata = item.metadata
            )
            for item in data
        ]
        embeddings = [item.embeddings for item in docs]

        # add to vector db
        vector_store.add_embeddings(texts=docs, embeddings=embeddings)

        # Persist store
        faiss.write_index(vector_store.index, VECTOR_DB_PATH)
        with open(DOCS_STORE_PATH, "wb") as f:
            pickle.dump(vector_store.docstore._dict, f)

        return {"status": "success", "stored": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_query(request: QueryRequest):
    try:
        query_embedding = embedding_model.embed_query(request.query)
        docs_and_scores = vector_store.similarity_search_by_vector(
            embedding=query_embedding,
            k=request.top_k
        )
        return [
            {
                "content": doc.page_content,
                "metadata": doc.metadata
            } for doc in docs_and_scores
        ]
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8001)

