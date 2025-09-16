import os
import asyncio
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv

# Hide TensorFlow oneDNN logs & warnings
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate

# ---- Event loop fix for Streamlit async ----
try:
    asyncio.get_running_loop()
except RuntimeError:
    asyncio.set_event_loop(asyncio.new_event_loop())

# ---- Load API key ----
load_dotenv()
with open("gemini key.txt") as f:
    key = f.read().strip()

genai.configure(api_key=key)
os.environ["GOOGLE_API_KEY"] = key

# ---- Streamlit UI ----
st.title("RAG Customer Support Chatbot")

# ---- Load PDF ----
loader = PyPDFLoader("user manual 1.pdf")
data = loader.load()

# ---- Split text ----
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(data)

# ---- Create embeddings ----
embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

# ---- Create / persist Chroma vectorstore ----
docsearch = Chroma.from_documents(
    docs,
    embedding,
    persist_directory="./chroma_db"
)
docsearch.persist()

retriever = docsearch.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 5}
)

# ---- LLM ----
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash-8b",
    temperature=0.8,
)

# ---- Prompt ----
system_prompt = (
    "You are a helpful e-commerce customer support assistant. "
    "Answer **only** from the provided context. "
    "If the answer isn't in context, say you don't have that info. "
    "Write concise, friendly answers. If relevant, include steps or timelines."
    "\n\n{context}"
)

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)

# ---- Query Handling ----
query = st.chat_input("Ask Question: ")
if query:
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    response = rag_chain.invoke({"input": query})

    st.write(response["answer"])
