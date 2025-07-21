from flask import Flask, render_template, jsonify, request
from src.helper import download_hugging_face_embeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq # Changed from langchain_openai
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
from src.prompt import * # Assuming system_prompt is defined here
import os


app = Flask(__name__)


load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
# Changed to GROQ_API_KEY
GROQ_API_KEY = os.environ.get('GROQ_API_KEY')

# Set environment variables for LangChain to pick them up
os.environ["PINECONE_API_KEY"] = PINECONE_API_KEY
os.environ["GROQ_API_KEY"] = GROQ_API_KEY # Changed to GROQ_API_KEY


# Ensure embeddings are downloaded/initialized
# This function should return an embeddings object compatible with PineconeVectorStore
embeddings = download_hugging_face_embeddings()

index_name = "medical-chatbot" # Ensure this index exists in your Pinecone account

# Connect to the existing Pinecone index
docsearch = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
)

# Create the retriever from the Pinecone vector store
retriever = docsearch.as_retriever(search_type="similarity", search_kwargs={"k":3})

# Initialize ChatGroq instead of ChatOpenAI
# You can choose a different model if needed, e.g., "llama3-70b-8192"
chatModel = ChatGroq(model_name="llama3-8b-8192", temperature=0)

# The prompt template remains the same
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt), # Assuming system_prompt is defined in src.prompt
        ("human", "{input}"),
    ]
)

# Create the question-answer chain
question_answer_chain = create_stuff_documents_chain(chatModel, prompt)

# Create the RAG chain
rag_chain = create_retrieval_chain(retriever, question_answer_chain)


@app.route("/")
def index():
    return render_template('chat.html')


@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input_text = msg # Renamed 'input' to 'input_text' to avoid conflict with built-in 'input'
    print(input_text)
    response = rag_chain.invoke({"input": input_text})
    print("Response : ", response["answer"])
    return str(response["answer"])


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)