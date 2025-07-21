# streamlit_app.py

import streamlit as st
import os
from dotenv import load_dotenv

# LangChain and Pinecone imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec

# LangChain RAG components
from langchain_groq import ChatGroq
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# --- Configuration ---
load_dotenv()

PINECONE_API_KEY = "YOUR_API_KEY"
GROQ_API_KEY = "YOUR_API_KEY"

if not PINECONE_API_KEY or not GROQ_API_KEY:
    st.error("Missing API keys. Please check your .env file.")
    st.stop()

os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# Config values
INDEX_NAME = "gitaverse"
embedding_model_name = "sentence-transformers/all-MiniLM-L6-v2"
groq_model_name = "llama3-8b-8192"
groq_temperature = 0

pdf_file_path = r"C:\Users\divis\OneDrive\Documents\GITAVERSE\Data\Gita.pdf"

# --- RAG Pipeline Setup Functions ---

@st.cache_resource
def get_pinecone_client():
    try:
        pinecone_instance = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_API_KEY)
        st.success("Pinecone initialized successfully.")
        return pinecone_instance
    except Exception as e:
        st.error(f"Error initializing Pinecone: {e}")
        st.stop()

@st.cache_resource
def get_embedding_model():
    try:
        embeddings = HuggingFaceEmbeddings(
            model_name=embedding_model_name,
            model_kwargs={'device': 'cpu'},
            encode_kwargs={'normalize_embeddings': False}
        )
        st.success(f"Embedding model '{embedding_model_name}' loaded.")
        return embeddings
    except Exception as e:
        st.error(f"Error loading embedding model: {e}")
        st.stop()

@st.cache_resource
def load_and_process_pdf(file_path, _embeddings_model, _pinecone_client_instance):
    st.info(f"Checking Pinecone index '{INDEX_NAME}' and loading PDF data...")
    try:
        if INDEX_NAME not in _pinecone_client_instance.list_indexes().names():
            st.warning(f"Index '{INDEX_NAME}' not found. Creating and uploading data...")

            loader = PyPDFLoader(file_path)
            documents = loader.load()
            if not documents:
                st.error("No documents loaded from PDF.")
                st.stop()

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                separators=["\n\n", "\n", " ", ""]
            )
            texts = text_splitter.split_documents(documents)
            if not texts:
                st.error("No text chunks generated from PDF.")
                st.stop()

            embedding_dimension = 384
            _pinecone_client_instance.create_index(
                name=INDEX_NAME,
                dimension=embedding_dimension,
                metric='cosine',
                spec=ServerlessSpec(cloud='aws', region='us-west-2')
            )
            st.success(f"Index '{INDEX_NAME}' created. Uploading {len(texts)} documents...")

            docsearch = PineconeVectorStore.from_documents(
                texts,
                _embeddings_model,
                index_name=INDEX_NAME
            )
            st.success("Documents uploaded to Pinecone.")
            return docsearch
        else:
            st.success(f"Connecting to existing Pinecone index: '{INDEX_NAME}'.")
            docsearch = PineconeVectorStore.from_existing_index(
                index_name=INDEX_NAME,
                embedding=_embeddings_model
            )
            return docsearch
    except Exception as e:
        st.error(f"Error during PDF processing or Pinecone interaction: {e}")
        st.stop()

# ✅ FIXED: use `_doc_store` so Streamlit doesn't try to hash it
@st.cache_resource
def get_rag_chain(_doc_store):
    st.info("Setting up RAG chain with Groq LLM...")
    llm = ChatGroq(temperature=groq_temperature, model_name=groq_model_name)

    system_prompt = (
        "You are a spiritual guru who is well versed with the Bhagavad Gita. "
        "Use the following pieces of retrieved context from the Gita to answer "
        "the question, reflecting a deep understanding of its teachings. "
        "If the answer is not found in the provided context, state that the answer "
        "is beyond the current scope of the provided text. Keep your answers "
        "concise and insightful, drawing wisdom from the Gita as appropriate.\n\n"
        "{context}"
    )

    prompt = ChatPromptTemplate.from_messages(
        [("system", system_prompt), ("human", "{input}")]
    )

    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = _doc_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    rag_chain = create_retrieval_chain(retriever, document_chain)
    st.success("RAG chain setup complete.")
    return rag_chain

# --- Streamlit UI ---

st.set_page_config(page_title="Gita Spiritual Guru Chatbot", layout="centered")
st.title("🕉️ GitaVerse")
st.markdown("Ask me anything about the Bhagavad Gita!")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- Main Logic ---

pinecone_client = get_pinecone_client()
embeddings_model = get_embedding_model()
docsearch_store = load_and_process_pdf(pdf_file_path, embeddings_model, pinecone_client)

# ✅ FIXED: pass `docsearch_store` as `_doc_store`
rag_chain = get_rag_chain(_doc_store=docsearch_store)

if prompt := st.chat_input("What is your spiritual question?"):
    st.chat_message("user").markdown(prompt)
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.spinner("Meditating on your question..."):
        try:
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
        except Exception as e:
            answer = f"An error occurred while seeking wisdom: {e}"
            st.error(answer)

    with st.chat_message("assistant"):
        st.markdown(answer)
    st.session_state.messages.append({"role": "assistant", "content": answer})
