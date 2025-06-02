import os
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.text_splitter import CharacterTextSplitter
from langchain_groq import ChatGroq

# === Load env vars ===
load_dotenv()
groq_api_key = os.environ.get("GROQ_API_KEY")

# === Set correct lightweight Groq model ===
llm = ChatGroq(api_key=groq_api_key, model_name="llama3-8b-8192")

# === Use smaller HuggingFace embedding model ===
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# === Chunk splitter ===
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# === Path for persistent FAISS ===
VECTOR_DB_PATH = "db/faiss_index"

def create_or_load_vectorstore(text=None):
    if os.path.exists(VECTOR_DB_PATH):
        return FAISS.load_local(VECTOR_DB_PATH, embeddings)

    if not text:
        raise ValueError("No text provided and no FAISS index found.")

    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    vector_store.save_local(VECTOR_DB_PATH)
    return vector_store

def get_chain():
    return load_qa_chain(llm, chain_type="stuff")

def get_answer(vector_store, chain, question, context=None):
    combined_query = f"{context} {question}" if context else question
    docs = vector_store.similarity_search(combined_query, k=5)

    input_question = f"Context: {context}\nQuestion: {question}" if context else question
    return chain.run(input_documents=docs, question=input_question)
