# GitaVerse 🕉️

**Ask me anything about the Bhagavad Gita!**  
A spiritual chatbot built with Streamlit, LangChain, Groq LLaMA3‑8B, and Pinecone—delivering context-aware wisdom from the Gita via RAG.

---


## 🔧 Tech Stack

- 🔹 **Frontend**: Streamlit
- 🔹 **PDF Loader**: `PyPDFLoader`
- 🔹 **Text Splitting**: `RecursiveCharacterTextSplitter`
- 🔹 **Embeddings**: `sentence-transformers/all-MiniLM-L6-v2`
- 🔹 **Vector DB**: Pinecone (Serverless)
- 🔹 **LLM**: Groq LLaMA3‑8B via `langchain_groq`

---

## 🚀 Features

1. **PDF ingestion**: Auto‐chunks and indexes the Gita PDF into Pinecone  
2. **RAG query pipeline**: Retrieves top-K context chunks  
3. **LLM-powered response**: Expert spiritual guidance from the Gita  
4. **Chat UI**: Powered by Streamlit’s `st.chat_message` interface  
5. **Session history**: Conversational context stored via `st.session_state`

---

## ⚙️ Quick Start

### Prerequisites

- Python 3.10+
- Access to Pinecone and Groq
- `Gita.pdf` in your local path

### Install

```bash
git clone https://github.com/yourusername/GitaVerse.git
cd GitaVerse
pip install -r requirements.txt


