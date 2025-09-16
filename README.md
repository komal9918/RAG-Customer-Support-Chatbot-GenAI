# 🧠 Customer Support Chatbot with RAG

## 📌 Overview
Customer support teams often deal with **repetitive queries** from customers, such as product details, policies, and FAQs.  
Handling these queries manually is **time-consuming, costly, and prone to delays**.  

This project solves that problem by building a **Customer Support Chatbot** using **Retrieval-Augmented Generation (RAG)**.  
The chatbot retrieves relevant knowledge from company documents  and combines it with a **Generative AI model** to provide **accurate, context-aware, and conversational responses**.  

---

## 🎯 Business Problem
- Customers face delays in receiving responses from human agents.  
- Agents spend too much time answering repetitive questions.  
- Knowledge is scattered across multiple documents, making it hard to search manually.  

## ✅ Solution
- Automate query handling with an **AI-powered chatbot**.  
- Use **LangChain** to connect a **retrieval system (ChromaDB)** with a **Generative AI model**.  
- Provide **instant, accurate answers** by retrieving context from documents and generating human-like responses.  
- Reduce support costs and improve **customer satisfaction**.  

---

## 🏗️ Tech Stack
- **Python** 🐍  
- **LangChain** → Framework for chaining LLMs and retrievals  
- **Google Generative AI (Gemini )** → For natural language responses  
- **ChromaDB** → Vector database for storing and retrieving document embeddings  
- **SentenceTransformers** → To generate document embeddings  
- **PyPDFLoader** → For loading PDF files  
- **Jupyter Notebook** → Development and testing  

---

## 🔄 Project Workflow
1. **Data Ingestion**
   - Load PDF documents using `PyPDFLoader`.
   - Store them in a structured format.

2. **Text Preprocessing**
   - Split text into manageable chunks using `RecursiveCharacterTextSplitter`.
3. **Embedding Generation**
   - Convert text chunks into embeddings using `SentenceTransformerEmbeddings`.
4. **Vector Storage**
   - Store embeddings in **ChromaDB** for efficient retrieval.
5. **Retrieval-Augmented Generation**
   - Fetch relevant documents from the vector store.
   - Pass them along with the user’s query into a **LangChain Retrieval Chain**.
6. **Response Generation**
   - Use **Google Generative AI** (`ChatGoogleGenerativeAI`) to generate a context-aware answer.
   - Return the response in a conversational format.

---

## 📂 Project Structure

📁 customer-support-chatbot-rag
│── customer_support_chatbot_rag.ipynb # Main notebook
│── README.md # Project documentation
│── requirements.txt # Dependencies
│── 📁 data/ # PDF documents
│── 📁 vectorstore/ # Persistent ChromaDB storage


---

## ⚙️ How To Run
1. **Clone the repository**
   ```bash
   git clone https://github.com/komal9918/RAG-Customer-Support-Chatbot-GenAI/blob/main/customer_support_chatbot_rag.ipynb

