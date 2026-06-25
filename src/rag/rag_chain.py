# src/rag/rag_chain.py

from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv

load_dotenv()

def create_llm():
    """Groq LLM load karo"""
    print("Groq LLM load ho raha hai...")
    llm = ChatGroq(
        model="llama3-8b-8192",
        temperature=0.3,
        api_key=os.getenv("GROQ_API_KEY")
    )
    print("LLM ready!")
    return llm

def create_rag_chain(vector_db):
    """RAG chain banana"""
    llm = create_llm()
    
    prompt_template = """
    Neeche diye gaye context ko padh ke question ka answer do.
    Agar answer context mein nahi hai toh 'Mujhe is baare mein 
    information nahi hai' bolo.
    
    Context: {context}
    Question: {question}
    
    Answer:"""
    
    PROMPT = PromptTemplate(
        template=prompt_template,
        input_variables=["context", "question"]
    )
    
    rag_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vector_db.as_retriever(
            search_kwargs={"k": 3}
        ),
        chain_type_kwargs={"prompt": PROMPT},
        return_source_documents=True
    )
    print("RAG Chain ready!")
    return rag_chain