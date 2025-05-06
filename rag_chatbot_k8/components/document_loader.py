import os
import glob
import pickle
import numpy as np

from sklearn.metrics.pairwise import cosine_similarity
from langchain_huggingface import HuggingFaceEmbeddings

class DocumentLoader:
    def __init__(self, docs_folder, filename_embeddings_path, embedding_model):
        self.docs_folder = docs_folder
        self.filename_embeddings_path = filename_embeddings_path
        self.embedding_model = embedding_model
    
    def load_md_files(self):
        """ Load all markdown files from the specified folder. """
        md_files = []
        for filepath in glob.glob(f"{self.docs_folder}/**/*.md", recursive=True):
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
                md_files.append({
                    'filename': os.path.basename(filepath),
                    'content': text
                })
        print(f"Loaded {len(md_files)} markdown files....")
        return md_files
    
    def precompute_filename_embeddings(self):
        """ Precompute embeddings for filenames and save them. """
        print('Precomputing filename embeddings...')
        md_files = self.load_md_files()
        filename_embeddings = []
        for doc in md_files:
            embeddings = self.embedding_model.embed_query(doc['filename'])
            filename_embeddings.append(embeddings)
        print(f"Precomputed {len(filename_embeddings)} filename embeddings....")
        with open(self.filename_embeddings_path, 'wb') as f:
            pickle.dump({
                'files': md_files,
                'embeddings': filename_embeddings
            }, f)
        return md_files, filename_embeddings
    
    def select_relevant_files(self, query, top_k):
        """ Select top-k relevant files based on query similarity. """
        if not os.path.exists(self.filename_embeddings_path):
            print('No filename embeddings found, precomputing....')
            md_files, filename_embeddings = self.precompute_filename_embeddings()
        else:
            print('Loading exisiting filename embeddings....')
            with open(self.filename_embeddings_path, 'rb') as f:
                data = pickle.load(f)
                md_files, filename_embeddings = data['files'], data['embeddings']
        
        query_embedding = self.embedding_model.embed_query(query)
        query_embedding_array = np.array(query_embedding).reshape(1, -1)
        filename_embeddings_array = np.array(filename_embeddings)

        similarities = cosine_similarity(query_embedding_array, filename_embeddings_array)[0]
        top_indices = similarities.argsort()[-top_k:][::-1]
        selected_files = [md_files[idx] for idx in top_indices]
        return selected_files