# 📘 RAG-Pipeline – Policy QA Assistant

An AI-powered **Policy Question Answering** system built with **Retrieval-Augmented Generation (RAG)**.  
It ingests PDFs/text, splits into chunks, embeds with SentenceTransformers, stores in **ChromaDB**, and retrieves the most relevant passages for a query.

This repo includes a working **Jupyter/VS Code Notebook** flow _and_ a FastAPI service scaffold you can enable later.

---

## ✨ Features

- 📄 **Ingestion (Notebook)**: load PDFs and text files with LangChain loaders  
- ✂️ **Chunking**: `RecursiveCharacterTextSplitter` (size 800, overlap 120)  
- 🔢 **Embeddings**: `SentenceTransformer("all-MiniLM-L6-v2")`  
- 🗃️ **Vector Store**: Persistent **Chroma** collection (`pdf_documents`)  
- 🔎 **Retriever**: query → top-k similarity search (returns metadata + scores)  
- 🧪 **Reproducible notebook**: the JSON below shows the exact cells you ran (with sample outputs)

---

## 🧱 Tech Stack

- Python 3.11  
- LangChain (core + community loaders)  
- SentenceTransformers (`all-MiniLM-L6-v2`)  
- ChromaDB (persistent)  
- scikit-learn (similarity utils)  
- (Optional) FastAPI + Uvicorn for API

---

## 📂 Project Structure (suggested)

app/
main.py # (optional) FastAPI endpoints: /ingest, /qa
embeddings.py # EmbeddingManager
vectorstore.py # VectorStore (Chroma)
retriever.py # RAGRetriever
utils.py # loaders & helpers
data/
pdf/ # put your PDFs here
text_files/ # sample .txt files
