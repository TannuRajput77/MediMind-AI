# src/rag/embeddings.py

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os

def create_embeddings():
    """Embedding model load karo"""
    print("Embedding model load ho raha hai...")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    print("Embedding model ready!")
    return embeddings

def create_vector_db(chunks, embeddings):
    """FAISS vector database banana"""
    print("Vector DB ban raha hai...")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # Local mein save karo
    vector_db.save_local("faiss_index")
    print("Vector DB save ho gayi!")
    return vector_db

def load_vector_db(embeddings):
    """Saved vector DB load karo"""
    vector_db = FAISS.load_local(
        "faiss_index",
        embeddings,
        allow_dangerous_deserialization=True
    )
    print("Vector DB load ho gayi!")
    return vector_db