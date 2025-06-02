import os
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from langchain.text_splitter import CharacterTextSplitter
# from langchain.llms.groq import ChatGroq
from langchain_groq import ChatGroq
from dotenv import load_dotenv

load_dotenv()

groq_api_key = os.environ.get("GROQ_API_KEY")
llm = ChatGroq(api_key=groq_api_key, model_name="all-MiniLM-L6-v2")
embeddings = HuggingFaceEmbeddings()

text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

def create_qa_pipeline(text):
    chunks = text_splitter.split_text(text)
    vector_store = FAISS.from_texts(chunks, embeddings)
    chain = load_qa_chain(llm, chain_type="stuff")
    return vector_store, chain


def get_answer(vector_store, chain, question, context=None):
    # If there is follow-up context, append it to the question for similarity search
    if context:
        combined_query = context + " " + question
    else:
        combined_query = question

    docs = vector_store.similarity_search(combined_query, k=5)  # increased k for more context

    # If you want, you can tweak chain.run to include context as extra param or in question itself
    input_question = question
    if context:
        input_question = f"Context: {context}\nQuestion: {question}"

    return chain.run(input_documents=docs, question=input_question)
