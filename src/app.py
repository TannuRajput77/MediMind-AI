# app.py
import streamlit as st
from langchain_groq import ChatGroq
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import tempfile
import os
from dotenv import load_dotenv

load_dotenv()

# Page config
st.set_page_config(
    page_title="MediMind AI",
    page_icon="🏥",
    layout="wide"
)

st.title("🏥 MediMind AI")
st.caption("Upload karo medical PDF aur koi bhi sawaal poochho!")

# Session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "vector_db" not in st.session_state:
    st.session_state.vector_db = None

# Sidebar
with st.sidebar:
    st.header("📂 PDF Upload")
    
    groq_key = st.text_input(
        "Groq API Key", 
        type="password",
        placeholder="gsk_..."
    )
    
    uploaded_file = st.file_uploader(
        "Medical PDF choose karo", 
        type="pdf"
    )
    
    if uploaded_file and groq_key:
        if st.button("🔄 Process PDF", type="primary"):
            with st.spinner("PDF process ho rahi hai..."):
                # Temp file mein save karo
                with tempfile.NamedTemporaryFile(
                    delete=False, suffix=".pdf"
                ) as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name
                
                # Load + split
                loader = PyPDFLoader(tmp_path)
                documents = loader.load()
                
                splitter = RecursiveCharacterTextSplitter(
                    chunk_size=500, 
                    chunk_overlap=50
                )
                chunks = splitter.split_documents(documents)
                
                # Embeddings + FAISS
                embeddings = HuggingFaceEmbeddings(
                    model_name="sentence-transformers/all-MiniLM-L6-v2"
                )
                st.session_state.vector_db = FAISS.from_documents(
                    chunks, embeddings
                )
                
                os.unlink(tmp_path)
                st.success(f"✅ {len(chunks)} chunks ready!")

# Chat interface
if st.session_state.vector_db is None:
    st.info("👈 Pehle sidebar mein PDF upload karo!")
else:
    # Purane messages dikhao
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # User input
    if question := st.chat_input("Koi bhi medical sawaal poochho..."):
        st.session_state.messages.append(
            {"role": "user", "content": question}
        )
        with st.chat_message("user"):
            st.write(question)
        
        with st.chat_message("assistant"):
            with st.spinner("Soch raha hoon..."):
                llm = ChatGroq(
                    model="llama-3.1-8b-instant",
                    temperature=0.3,
                    api_key=groq_key
                )
                
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": 3}
                )
                
                def format_docs(docs):
                    return "\n\n".join(
                        doc.page_content for doc in docs
                    )
                
                prompt = PromptTemplate.from_template(
                    "Context: {context}\nQuestion: {question}\nAnswer:"
                )
                
                chain = (
                    {
                        "context": retriever | format_docs,
                        "question": RunnablePassthrough()
                    }
                    | prompt | llm | StrOutputParser()
                )
                
                answer = chain.invoke(question)
                st.write(answer)
                
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer}
                )