from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document


class TextSplitter:
    def __init__(self, chunk_size=1500, chunk_overlap=50):
        self.text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    def split_documents(self, md_docs):
        """Split markdown documents into chunks."""
        print('Splitting documents into chunks...')
        documents = []
        for doc in md_docs:
            split_texts = self.text_splitter.split_text(doc['content'])
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