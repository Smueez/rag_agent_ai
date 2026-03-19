# 🚀 Agentic RAG System with Self-Reflection (Qdrant + OpenAI)

A production-grade Retrieval-Augmented Generation (RAG) system built with an **agentic architecture**, featuring:

- 🔍 Semantic search with vector database  
- 🧠 Multi-agent reasoning pipeline  
- 🔁 Self-reflection (critic loop)  
- 📊 Re-ranking and grounded answer generation  
- 🌍 Multilingual embeddings support  

---

# 📌 Overview

This project implements a modern **Agentic RAG pipeline** where multiple AI agents collaborate to:

1. Understand the user query  
2. Retrieve relevant knowledge  
3. Re-rank and filter context  
4. Generate grounded answers  
5. Critically evaluate and refine responses  

The goal is to **minimize hallucination** and **maximize factual correctness**.

---

## 🧠 Architecture

```
User Query → FastAPI API → Planner Agent
                              ↓
                      Query Classify Agent
                              ↓                        ↓
                      Query Re-Write Agent
                              ↓
                 Retriever Tool (Vector Search)
                              ↓
                       Re-Ranking Agent
                              ↓
                    Answer Generator Agent
                              ↓
                        Grounding Agent
                              ↓
                 Critic Agent (Self-Reflection)
                              ↓
                        Final Response
```

---

# ⚙️ Tech Stack

- LLM: OpenAI (GPT-4o / GPT-4o-mini)  
- Embeddings: Any Sentence Transformer (multilingual-e5-base)  
- Vector DB: Qdrant  
- Backend: FastAPI  
- Orchestration: OpenAI Agents SDK  
- Language: Python  

---
# 🔑 Key Features

## 1. Agentic RAG (Not Traditional RAG)

Unlike simple pipelines, this system uses multiple agents:

- Planner Agent → Calls many other agentic tools and returns response and also works as an orchestration layer.
- Query Classify Agent → Classify the user's input, and classify if the input is a valid query or just some random conversation! 
- Query Rewrite Agent → Rewrites the user's query in more contextual way. 
- Re-ranking Agent → improves retrieval quality  
- Answer Generator → produces grounded response  
- Grounding → Ground the output 
- Critic Agent → validates and refines  

---

## 2. Self-Reflection Loop (Critic Agent)


The system evaluates its own output:
```commandline
Draft Answer
↓
Critic Evaluation
↓
Revise (if needed)
```

This reduces:
- hallucination  
- incomplete answers  
- incorrect claims  

---

## 3. Re-Ranking for Better Retrieval

Vector similarity alone is not enough.

We improve results by:
- re-ranking retrieved chunks using LLM reasoning  
- filtering irrelevant context  
- prioritizing high-quality chunks  

---

## 4. Grounded Answer Generation

The answer generator strictly follows:

- uses ONLY retrieved context  
- avoids external knowledge  
- returns fallback if context is insufficient  

---


---

# 🔄 Pipeline Breakdown

## 1. Document Processing
```
PDF → Chunking → Embedding → Vector DB
```
## 2. Retrieval
```
Query → Embedding → Vector Search → Top-K Chunks
```
## 🚀 Quick Start

### 1. Clone and Setup
```bash
# Create project directory
mkdir rag-agent-project
cd rag-agent-project

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment

Create a `.env` file:
```env
HF_AUTH_KEY=your-hf-key
OPEN_AI_API_KEY=your-api-key
EMBEDDING_MODEL=your-sentence-transformer-model

MAX_TOKEN=8191
# Qdrant Configuration
QDRANT_HOST=localhost
QDRANT_PORT=6333
QDRANT_COLLECTION_NAME=document_embeddings
VECTOR_SIZE=768
# RAG Configuration
CHUNK_SIZE=400
CHUNK_OVERLAP=80
TOP_K_RESULTS=5
SIMILARITY_THRESHOLD=0.7

# Agent Configuration
MAX_ITERATIONS=3
ENABLE_SELF_REFLECTION=true

# API Configuration
API_HOST=0.0.0.0
API_PORT=8000
```

### 4. Start Qdrant Vector Database
```bash
docker compose -f docker-compose-infra.yml up --build
```

Verify Qdrant is running:
```bash
curl http://localhost:6333/
```
### 5. Upload and Store your pdf files
```bash
python process_n_store_service.py <path_to_pdf>
```
### 5. Start the API Server
```bash
uvicorn app.main:app --reload
```

The API will be available at: `http://localhost:8000`

API Documentation: `http://localhost:8000/docs`
