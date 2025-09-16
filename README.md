# 🤖 Customer Support Chatbot using Retrieval-Augmented Generation (RAG)

## 📌 Overview
Customer support teams often handle thousands of **repetitive queries** related to orders, returns, payments, warranties, and policies.  
This project solves that by building an **AI-powered chatbot** that retrieves information from company manuals (e.g., **user manuals, FAQs, policies**) and generates accurate, conversational responses.  

The chatbot is built with **LangChain, ChromaDB, and Google Generative AI (Gemini)**, and deployed using **Streamlit** for a web-based interface.  

---

## 🎯 Business Problem
- Customers wait too long for responses from human agents.  
- Agents repeatedly answer the same FAQs.  
- Information is scattered across manuals and documents, making it hard to search.  

## ✅ Solution
- Automate query resolution with an **AI chatbot**.  
- Use **Retrieval-Augmented Generation (RAG)** to combine **document retrieval (ChromaDB)** with **LLM-powered generation (Gemini AI)**.  
- Provide **instant, reliable, and user-friendly answers**.  
- Improve customer satisfaction and reduce support costs.  

---

## 🚀 Features
- 📂 Load customer support manuals (PDF).  
- ✂️ Chunk documents into retrievable embeddings.  
- 🔍 Store and query with **Chroma vector database**.  
- 🤖 Answer user queries using **Google Gemini LLM**.  
- 🖥️ Interactive **Streamlit web interface**.  
- 📜 Includes FAQs, privacy policy, terms, refund policy, and shipping rules from the **user manual**.  

---

## 🏗️ Tech Stack
- **Python 3.10+**  
- **Streamlit** → Web app UI  
- **LangChain** → Orchestrates retrieval + generation  
- **ChromaDB** → Vector database for storing embeddings  
- **HuggingFace SentenceTransformers** → Text embeddings  
- **Google Generative AI (Gemini)** → Conversational response generation  
- **PyPDFLoader** → PDF ingestion  
- **Dotenv** → API key management  

---

## 🔄 Workflow
1. **Document Loading**  
   - Load `user manual 1.pdf` with `PyPDFLoader`.
2. **Add API Key
   - Add API Key using Gemini
3. **Text Splitting**  
   - Split into chunks with `RecursiveCharacterTextSplitter`.  

4. **Embeddings & Vector Store**  
   - Convert chunks to embeddings using `sentence-transformers/all-MiniLM-L6-v2`.  
   - Store vectors in **ChromaDB** for efficient retrieval.  

5. **Retriever + LLM**  
   - Retrieve top-k (k=5) most relevant chunks.  
   - Feed them into **Gemini model (`gemini-1.5-flash-8b`)**.  

6. **Chat Interface**  
   - Users type questions in Streamlit chat box.  
   - RAG chain generates friendly, context-aware answers.  

---

## 📂 Project Structure

📁 customer-support-chatbot-rag
│── app.py # Streamlit web app
│── customer_support_chatbot_rag.py # Jupyter/Colab version of chatbot
│── user manual 1.pdf # Source knowledge base (FAQs, policies)
│── gemini key.txt # Google API key (⚠️ keep private)
│── README.md # Project documentation
│── chroma_db/ # Vectorstore persistence (created after first run)

1. **Clone repository**
   ```bash
   git clone https://github.com/yourusername/customer-support-chatbot-rag.git
   cd customer-support-chatbot-rag
