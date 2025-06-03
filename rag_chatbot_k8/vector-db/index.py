from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
from typing import List
from langchain_community.vectorstores import FAISS
from langchain_openai import AzureOpenAIEmbeddings
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.schema import Document
import faiss
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Constants
VECTOR_STORE_PATH = "vector_store"
INDEX_NAME = "index"
EMBEDDING_DIM = 1536  # for Azure text-embedding-ada-002, adjust if needed
PORT = int(os.getenv("PORT", 8001))
HOST = os.getenv("HOST")


# Fast API Setup
app = FastAPI()

# enable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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
    top_k: int = 10


def load_vector_store():
    if os.path.exists(os.path.join(VECTOR_STORE_PATH, f"{INDEX_NAME}.faiss")):
        return FAISS.load_local(VECTOR_STORE_PATH, embedding_model, index_name=INDEX_NAME, allow_dangerous_deserialization=True)
    else:
        from faiss import IndexFlatL2
        index = IndexFlatL2(EMBEDDING_DIM)
        return FAISS(embedding_model, index, InMemoryDocstore({}), {})


vector_store = load_vector_store()


@app.post("/store")
async def store_embeddings(data: List[EmbeddingItem]):
    try:
        docs = [
            Document(
                page_content=item.content,
                metadata=item.metadata
            )
            for item in data
        ]

        # add to vector db
        vector_store.add_texts(
            texts=[doc.page_content for doc in docs],
            metadatas=[doc.metadata for doc in docs]
        )

        # Persist store
        vector_store.save_local(VECTOR_STORE_PATH, index_name=INDEX_NAME)

        return {"status": "success", "stored": len(data)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/search")
async def search_query(request: QueryRequest):
    try:
        print(request.query)
        docs_and_scores = vector_store.similarity_search(
            query=request.query,
            k=request.top_k
        )
        # print('docs-core: ', docs_and_scores) # Document schema 5 times since k= 5
        # Document(                                 
        #    id='5492c27c-f490-4bb5-b691-1877216adcd9', 
        #    metadata={'source': 'configmap.md-2'}, 
        #    page_content='Here\'s an example ConfigMap that has some keys ... like a fragment of a configuration\n')
        # )
        result = [{
                "content": doc.page_content,
                "metadata": doc.metadata
            } 
            for doc in docs_and_scores
        ]
        print("Sent result to query-backend")
        return result
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))



@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    uvicorn.run(app, host=HOST, port=PORT)

