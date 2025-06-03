import subprocess
import shutil
from pathlib import Path
import pickle
import os
import requests
import glob
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import AzureOpenAIEmbeddings
from dotenv import load_dotenv

#  for hash code
import hashlib
import json

HASH_DB_PATH = Path("hash_files.json")

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


app = FastAPI()

# enable cors
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def clone_or_pull_repo():
    if not TEMP_DIR.exists():
        print("✅ Cloning Kubernetes docs repo...")
        subprocess.run(["git", "clone", REPO_URL, str(TEMP_DIR)], check=True)
    else:
        print("✅ Pulling latest changes...")
        subprocess.run(["git", "-C", str(TEMP_DIR), "pull"], check=True)


def copy_docs():
    base_dir = TEMP_DIR / "content" / "en" / "docs"
    # selected_subdirs = ["concepts", "reference", "setup", "tasks"]
    selected_subdirs = ["concepts"]

    for subdir in selected_subdirs:
        source_subdir = base_dir / subdir
        if not source_subdir.exists():
            print(f"⚠️ Skipping missing subdir: {source_subdir}")
            continue

        print(f"✅ Copying docs from {source_subdir} to {TARGET_DIR}...")
        TARGET_DIR.mkdir(parents=True, exist_ok=True)

        for file in source_subdir.glob("**/*.md"):
            relative_path = file.relative_to(base_dir)
            dest_file = TARGET_DIR / relative_path
            dest_file.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(file, dest_file)
            print(f"📄 Copied: {file} -> {dest_file}")


def load_md_files():
    md_files = []
    try:
        for filepath in glob.glob(f"{TARGET_DIR}/concepts/**/*.md", recursive=True):
            print(filepath)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            md_files.append({
                'filename': os.path.basename(filepath),
                'content': text,
            })
    except Exception as e:
        print("Error reading files:", e)
    print(f"Loaded {len(md_files)} markdown files....")
    return md_files
        
def call_text_splitter(md_docs):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1500, chunk_overlap=50)

    documents = []
    for doc in md_docs:
        split_texts = text_splitter.split_text(doc['content'])
        for i, chunk in enumerate(split_texts):
            document = Document(
                page_content=chunk,
                metadata={
                    'source': f"{doc['filename']}-{i}" 
                }
            )
            documents.append(document)
    print(f"Total document chunks created: {len(documents)}")
    return documents

def get_file_hash(content):
    return hashlib.sha256(content.encode('utf-8')).hexdigest()

def load_existing_hashes():
    if HASH_DB_PATH.exists():
        with open(HASH_DB_PATH, "r") as f:
            return json.load(f)
    return {}

def save_hashes(hashes):
    with open(HASH_DB_PATH, "w") as f:
        json.dump(hashes, f, indent=2)

#  embeddings
def rerun_embeddings():
    """ Recompute filename embeddings and save them. """
    print("🔁 Recomputing text embeddings for filenames...")

    #  initalize hash
    existing_hashes = load_existing_hashes()
    print('')
    new_hashes = {}
    to_embed = []

    # Document Loading Process
    md_files = load_md_files()
    print(f"Done with loading .md-files: {len(md_files)}")
    # md_files = [{
    #     'filename': os.path.basename(filepath),
    #     'content': text,
    # }, ... ]

    for doc in md_files:
        filename = doc['filename']
        content_hash = get_file_hash(doc["content"])
        new_hashes[filename] = content_hash

        if existing_hashes.get(filename) != content_hash:
            print(f"📌 Change detected: {filename}")
            to_embed.append(doc)
        else:
            print(f"✅ No change: {filename}")

    if not to_embed:
        print("🎉 All documents are up-to-date.")
        return

    print('to_embed: ', to_embed[0])
    # # Text Splitting Process
    chunk_documents = call_text_splitter(to_embed)
    print(f"Done with splits: {len(chunk_documents)}")
    print(f"Done with splits: {chunk_documents[0]}")
    # # chunk_documents = Document(
    # #     page_content=chunk,
    # #     metadata={
    # #         'source': f"{doc['filename']}-{i}" 
    # #     }
    # # )  

    # # # Embedding Process
    payload = []
    for doc in chunk_documents:
        embedding = embedding_model.embed_query(doc.metadata['source'])
        payload.append({
            "embedding": embedding,
            "metadata": doc.metadata,
            "content": doc.page_content
        })
    print(f"✅ Computed {len(payload)} filename embeddings.")
    print(f"✅ Updated embeddings for {len(to_embed)} changed files.")

    # # POST all to vector store
    try:
        print(payload)
        response = requests.post("http://localhost:8001/store", json=payload)
        print(f"🧪 Response status: {response.status_code}")
        print(f"🧪 Response text: {response.text}")
        response.raise_for_status()
        print("✅ Successfully pushed embeddings to vector store.")
    except Exception as e:
        print(f"❌ Failed to push to vector store: {e}")

    # # Save file hashes after success
    save_hashes(new_hashes)    


if __name__ == "__main__":
    try:
        # clone_or_pull_repo()
        # copy_docs()
        rerun_embeddings()
        print("✅ Successfully stored embeddings... ")

    except Exception as e:
        print(f"❌ Error: {e}")
