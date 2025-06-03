import subprocess
import shutil
from pathlib import Path
import pickle
import os, requests
import time
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

from components.text_splitter import TextSplitter
from components.document_loader import DocumentLoader
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure retry strategy
retry_strategy = Retry(
    total=5,  # number of retries
    backoff_factor=1,  # wait 1, 2, 4, 8, 16 seconds between retries
    status_forcelist=[429, 500, 502, 503, 504]  # HTTP status codes to retry on
)
http_adapter = HTTPAdapter(max_retries=retry_strategy)
session = requests.Session()
session.mount("http://", http_adapter)
session.mount("https://", http_adapter)

# Azure embedding model
embedding_model = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_ENDPOINT"),
    deployment=os.getenv("AZURE_EMBEDDING_DEPLOYMENT"),
    api_key=os.getenv("AZURE_API_KEY"),
    openai_api_version=os.getenv("AZURE_EMBEDDING_VERSION")
)

# Constants
REPO_URL = "https://github.com/kubernetes/website"
TEMP_DIR = Path("./temp-docs")
TARGET_DIR = Path("./k8_docs/en/concepts")
FILENAME_EMBEDDINGS_PATH = Path("filename_embeddings.pkl")

def clone_or_pull_repo():
    if not TEMP_DIR.exists():
        print("✅ Cloning Kubernetes docs repo...")
        subprocess.run(["git", "clone", REPO_URL, str(TEMP_DIR)], check=True)
    else:
        print("✅ Pulling latest changes...")
        subprocess.run(["git", "-C", str(TEMP_DIR), "pull"], check=True)

def copy_docs():
    source_dir = TEMP_DIR / "content" / "en" / "docs"
    if not source_dir.exists():
        raise FileNotFoundError(f"❌ content/en/docs directory not found in {TEMP_DIR}")

    print(f"✅ Copying docs from {source_dir} to {TARGET_DIR}...")
    TARGET_DIR.mkdir(parents=True, exist_ok=True)

    for file in source_dir.glob("**/*.md"):
        relative_path = file.relative_to(source_dir)
        dest_file = TARGET_DIR / relative_path
        dest_file.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(file, dest_file)


def rerun_embeddings():
    """ Recompute filename embeddings and save them. """
    print("🔁 Recomputing text embeddings for filenames...")

    # Document Loading Process
    doc_loader = DocumentLoader(
        docs_folder=str(TARGET_DIR),
        filename_embeddings_path=FILENAME_EMBEDDINGS_PATH,
        embedding_model=embedding_model
    )

    md_files = doc_loader.load_md_files()
    print(f"Done with loading .md-files: {len(md_files)}")
    if md_files:
        print(f".md-files sample: {md_files[0]}")

    # Text Splitting Process
    text_splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunk_documents = text_splitter.split_documents(md_files)
    print(f"Done with splits: {len(chunk_documents)}")

    # Embedding Process
    try:
        # Process documents in smaller batches
        batch_size = 50  # Reduced batch size
        payload = []
        
        for i in range(0, len(chunk_documents), batch_size):
            batch = chunk_documents[i:i + batch_size]
            print(f"\nProcessing batch {i//batch_size + 1}/{(len(chunk_documents) + batch_size - 1)//batch_size}")
            
            contents = []
            for doc in batch:
                try:
                    contents.append(doc.page_content)
                except Exception as e:
                    print(f"Error accessing document content: {e}")
                    continue
            
            if not contents:
                continue
                
            try:
                embeddings = embedding_model.embed_documents(contents)
                
                for doc, embedding in zip(batch, embeddings):
                    try:
                        payload.append({
                            "embedding": embedding,
                            "metadata": doc.metadata,
                            "content": doc.page_content
                        })
                    except Exception as e:
                        print(f"Error creating payload item: {e}")
                        continue
                
                print(f"Processed {len(payload)} documents so far...")
                
                # Send batch to vector store
                if len(payload) >= batch_size:
                    print(f"Sending batch of {len(payload)} to vector store...")
                    response = session.post("http://localhost:8001/store", json=payload)
                    response.raise_for_status()
                    print("✅ Successfully pushed batch to vector store.")
                    payload = []  # Clear payload after successful send
                    
                    # Add delay between batches to respect rate limits
                    time.sleep(2)  # 2 second delay between batches
                    
            except Exception as e:
                print(f"Error processing batch: {e}")
                time.sleep(60)  # Wait 60 seconds on rate limit error
                continue
        
        # Send any remaining documents
        if payload:
            print(f"Sending final batch of {len(payload)} to vector store...")
            response = session.post("http://localhost:8001/store", json=payload)
            response.raise_for_status()
            print("✅ Successfully pushed final batch to vector store.")

    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")
        import traceback
        print("Traceback:")
        print(traceback.format_exc())
        raise

if __name__ == "__main__":
    try:
        # clone_or_pull_repo()
        # copy_docs()
        rerun_embeddings()
        print("✅ Successfully stored embeddings... ")

    except Exception as e:
        print(f"❌ Error: {e}")
