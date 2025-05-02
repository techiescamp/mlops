from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain.memory import ConversationBufferMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from .utils import clean_markdown

class RAGComponents:
    @staticmethod
    def retrieve_documents(vector_store, k=4):
        """ Create a retriever from the vector store. """
        return vector_store.as_retriever(search_kwargs={'k': k})
    
    @staticmethod
    def create_memory():
        """ Create a memory object for conversation history. """
        history = ChatMessageHistory()
        return ConversationBufferMemory(
            chat_memory=history,
            memory_key='chat_history',
            k=5,
            return_messages=True,
            output_key="answer"
        )
    
    @staticmethod
    def create_augmentation_chain(retriever, llm, memory):
        """ Create the RAG augmentation chain. """
        prompt_template = ChatPromptTemplate.from_template(
            """
            You are a helpful AI assistant that explain concepts to beginner. Use the following context and chat history to answer the question. Avoid spelling mistakes.
            If the context provided does NOT help answer the question, clearly mention that it's "out of context" and prefix your answer with a emoji symbol of your choice.

            Chat History: {chat_history}
            Context: {context}
            Question: {question}
            Answer: 
            """
        )

        def format_docs(docs):
            return "\n\n".join(clean_markdown(doc.page_content) for doc in docs)
        
        chain = (
            {
                "context": lambda x: format_docs(retriever.invoke(x)),
                "question": RunnablePassthrough(),
                "chat_history": lambda x: memory.load_memory_variables({"question": x})['chat_history'] 
            }
            | prompt_template
            | llm
            | StrOutputParser()
        ) 
        return chain