import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import AzureChatOpenAI
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
import tiktoken
import glob

# Load environment variables
load_dotenv()
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_API_KEY = os.getenv("AZURE_API_KEY")
AZURE_DEPLOYMENT = os.getenv("AZURE_DEPLOYMENT")
OPENAI_API_VERSION = os.getenv("OPENAI_API_VERSION")

# Azure AI configuration
llm = AzureChatOpenAI(
    azure_endpoint=AZURE_ENDPOINT,
    azure_deployment=AZURE_DEPLOYMENT,
    api_key=AZURE_API_KEY,
    openai_api_version=OPENAI_API_VERSION    
)

# Pricing for cost estimation
INPUT_TOKEN_COST = 0.00015 / 1000
OUTPUT_TOKEN_COST = 0.0006 / 1000

# Token counter
def count_tokens(text, model):
    encoding = tiktoken.encoding_for_model(model)
    return len(encoding.encode(text))


# Cost estimation
def estimate_cost(input_text, output_text):
    input_tokens = count_tokens(input_text, AZURE_DEPLOYMENT)
    output_tokens = count_tokens(output_text, AZURE_DEPLOYMENT)
    total_cost = (input_tokens * INPUT_TOKEN_COST) + (output_tokens * OUTPUT_TOKEN_COST)
    return input_tokens, output_tokens, total_cost

def load_md_files():
    md_files = []
    for filepath in glob.glob("docs/*.md"):
        with open(filepath, 'r', encoding='utf-8') as f:
            text = f.read()
            md_files.append({
                'filename': os.path.basename(filepath),
                'content': text
            })
    return md_files

# Split text into chunks
def text_splitter(md_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    documents = []
    for doc in md_docs:
        split_texts = text_splitter.split_text(doc['content'])
        for i, chunk in enumerate(split_texts):
            document = Document(
                page_content=chunk,
                metadata={"source": f"{doc['filename']}-chunk-{i}"}
            )
            documents.append(document)
    return documents

# Load or create vector store
def load_vector_store(embedding_model, documents):
    if os.path.exists('./faiss-db'):
        return FAISS.load_local('./faiss-db', embeddings=embedding_model, allow_dangerous_deserialization=True)
    else:
        vector_store = FAISS.from_documents(documents, embedding_model)
        vector_store.save_local('./faiss-db')
        return vector_store

# RAG components
def retrieve_documents(vector_store):
    return vector_store.as_retriever(search_kwargs={"k": 3})

def create_memory():
    return ConversationBufferMemory(
        memory_key="chat_history",
        k=5,
        return_messages=True,
        output_key="answer"
    )

def augmentation(retriever, memory):
    prompt_template = ChatPromptTemplate.from_template(
        """
        You are a helpful AI assistant. Use the following context and chat history to answer the question. 
        If you don't know the answer, respond that it is out of context and suggest querying technical experts.

        chat history: {chat_history}

        context: {context}

        question: {question}

        answer: 
        """
    )
    
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough(),
            "chat_history": lambda x: memory.load_memory_variables({"question": x})["chat_history"]
        }
        | prompt_template
        | llm
        | StrOutputParser()
    )
    return chain

#  FastAPI app
app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow frontend origin
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods (GET, POST, etc.)
    allow_headers=["*"],  # Allow all headers
)

# pydantic variables for RAG
class QueryRequest(BaseModel):
    query: str

# Global variables for RAG
embedding_model = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
md_docs = load_md_files()
documents = text_splitter(md_docs)
vector_store = load_vector_store(embedding_model, documents)
retriever = retrieve_documents(vector_store)
memory = create_memory()
augmentation_chain = augmentation(retriever, memory)

@app.post("/query")
async def query_rag(request: QueryRequest):
    print(request.query)
    try:
        query = request.query
        search_docs = retriever.invoke(query)
        context = "\n\n".join(doc.page_content for doc in search_docs)
        input_text = f"Chat History: {memory.load_memory_variables({})['chat_history']}\nContext: {context}\nQuestion: {query}"
        
        result = augmentation_chain.invoke(query)
        input_tokens, output_tokens, cost = estimate_cost(input_text, result)
        memory.save_context({"question": query}, {"answer": result})
        
        return {
            "answer": result,
            "sources": [doc.metadata['source'] for doc in search_docs],
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "estimated_cost": cost
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000) 