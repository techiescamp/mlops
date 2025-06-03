import subprocess
import shutil
from pathlib import Path
import pickle
import os, requests

from components.text_splitter import TextSplitter
from components.document_loader import DocumentLoader
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

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
TARGET_DIR = Path("./k8_docs/en")
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
    print(f".md-files: {md_files[0]}")
    # md_files = [{
    #     'filename': os.path.basename(filepath),
    #     'content': text,
    # }, ... ]

    # Text Splitting Process
    text_splitter = TextSplitter(
        chunk_size=1000,
        chunk_overlap=50
    )
    chunk_documents = text_splitter.split_documents(md_files)
    print(f"Done with splits: {len(chunk_documents)}")
    # chunk_documents = Document(
    #     page_content=chunk,
    #     metadata={
    #         'source': f"{doc['filename']}-{i}" 
    #     }
    # )

    # Embedding Process
    filename_embeddings = []
    for doc in chunk_documents:
        embedding = embedding_model.embed_query(doc['filename'])
        filename_embeddings.append(embedding)

    print(f"✅ Computed {len(filename_embeddings)} filename embeddings.")

    with open(FILENAME_EMBEDDINGS_PATH, 'wb') as f:
        pickle.dump({'files': md_files, 'embeddings': filename_embeddings}, f)

    print("📦 Saved embeddings to filename_embeddings.pkl.")
    return 
  

if __name__ == "__main__":
    try:
        # clone_or_pull_repo()
        # copy_docs()
        rerun_embeddings()
        print("✅ Successfully stored embeddings... ")

    except Exception as e:
        print(f"❌ Error: {e}")
