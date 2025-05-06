import os
from langchain_community.vectorstores import FAISS


class VectorStoreManager():
    def __init__(self, vector_store_path, embedding_model, batch_size=100):
        self.vector_store_path = vector_store_path
        self.embedding_model = embedding_model
        self.batch_size = batch_size

    
    def load_vector_store(self, documents):
        """Load vector store with embeddings."""
        print('Loading vector store...')
        vector_store = None

        if os.path.exists(self.vector_store_path):
            print('Loading existing FAISS...')
            vector_store = FAISS.load_local(
                self.vector_store_path, 
                embeddings=self.embedding_model,
                allow_dangerous_deserialization=True
            )
            # get existing document sources from metadata
            existing_sources = set(
                doc.metadata.get('source', '') for doc in vector_store.docstore._dict.values()
            )
            print('exisitng-docs-faiss: ', existing_sources.__len__())

            # identify new documents not already in the vector store
            new_documents = [
                doc for doc in documents
                if doc.metadata.get('source', '') not in existing_sources
            ]
            print('new-docs-faiss: ', new_documents.__len__())

            # add new documents to vector store
            if new_documents:
                print(f"Adding {len(new_documents)} new documents to vector_store")
                for i in range(0, len(new_documents), self.batch_size):
                    batch = new_documents[i:i+self.batch_size]
                    vector_store.add_documents(batch)
        else:
            print('Creating new FAISS DB...')
            for i in range(0, len(documents), self.batch_size):
                batch = documents[i : i+self.batch_size]
                if vector_store is None:
                    vector_store = FAISS.from_documents(documents, self.embedding_model)
                    print(f"FAISS DB created with {len(documents)} documents")
                else:
                    vector_store.add_documents(batch)
                    print(f"Added {len(batch)} documents to FAISS DB")
        vector_store.save_local(self.vector_store_path)
        return vector_store

        