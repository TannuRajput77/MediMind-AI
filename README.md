# 🏥 MediMind AI

A healthcare chatbot I built that can read any medical PDF and answer questions from it — like having a doctor's assistant that reads documents for you.

## Why I Built This
Most AI chatbots only know what they were trained on. I wanted to build something that could read *new* documents and answer from them in real-time. That's where RAG came in.

## How It Works
1. You upload any medical PDF
2. The system breaks it into chunks and converts them to vectors
3. When you ask a question, it finds the most relevant chunks
4. Groq's LLaMA model reads those chunks and gives you a precise answer

## Tech I Used
- **LangChain** — to build the RAG pipeline
- **FAISS** — Facebook's vector database for fast similarity search
- **HuggingFace** — sentence-transformers for creating embeddings
- **Groq (LLaMA 3.1)** — fast and free LLM for generating answers
- **Streamlit** — for building the chat UI quickly

## Challenges I Faced
- LangChain had breaking changes between versions — debugged import errors
- HuggingFace models needed offline caching to avoid rate limits
- Groq deprecated llama3-8b so switched to llama-3.1-8b-instant

These were actually great learnings about working with production AI systems.

## What's Next
- Add disease prediction ML models
- Support multiple PDFs at once
- Deploy on AWS EC2 with Docker

## Run It Yourself
```bash
git clone https://github.com/TannuRajput77/MediMind-AI.git
cd MediMind-AI
python -m venv medimind_env
medimind_env\Scripts\activate
pip install -r requirements.txt
streamlit run src/app.py
