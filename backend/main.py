"""
Day 7: FastAPI RAG Backend
Project: Nepali Knowledge Assistant
--------------------------------------
POST /ask  → embed query → FAISS search → build prompt → Ollama → return answer

Run with:
    uvicorn main:app --reload --port 8000

Test with:
    curl -X POST http://localhost:8000/ask \
         -H "Content-Type: application/json" \
         -d "{\"question\": \"नेपालको राजधानी कुन हो?\"}"
"""

import json
import os
import numpy as np
import faiss
import requests
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer

# ──────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────
MODEL_NAME  = "paraphrase-multilingual-mpnet-base-v2"
OLLAMA_URL  = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"
INDEX_FILE  = "data/faiss.index"
MAP_FILE    = "data/chunk_map.json"
TOP_K       = 5      # number of chunks to retrieve
DEVICE      = "cpu" # change to "cpu" if needed


# ──────────────────────────────────────────
# STARTUP — load everything once into memory
# ──────────────────────────────────────────
print("\n Loading models and index...")

embed_model = SentenceTransformer(MODEL_NAME, device=DEVICE)
print("  Embedding model loaded")

index = faiss.read_index(INDEX_FILE)
print(f"  FAISS index loaded ({index.ntotal} vectors)")

with open(MAP_FILE, "r", encoding="utf-8") as f:
    chunk_map = json.load(f)
print(f"  Chunk map loaded ({len(chunk_map)} chunks)")

print("  Ready.\n")


# ──────────────────────────────────────────
# FASTAPI APP
# ──────────────────────────────────────────
app = FastAPI(title="Nepali Knowledge Assistant", version="0.1.0")

# Allow React frontend (Day 8) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:8000"],
    allow_methods=["*"],
    allow_headers=["*"],
)



class AskRequest(BaseModel):
    question: str

class Source(BaseModel):
    title: str
    url:   str
    chunk_text: str
    score: float

class AskResponse(BaseModel):
    answer:   str
    sources:  list[Source]
    question: str


def retrieve(question: str, k: int = TOP_K) -> list[dict]:
    """Embed the question and retrieve top-k chunks from FAISS."""
    q_vec = embed_model.encode(
        [question],
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype("float32")

    scores, ids = index.search(q_vec, k)

    results = []
    for score, idx in zip(scores[0], ids[0]):
        if idx == -1:           # FAISS returns -1 if not enough vectors
            continue
        chunk = chunk_map[idx]
        results.append({
            "title":      chunk["title"],
            "url":        chunk["url"],
            "chunk_text": chunk["chunk_text"],
            "score":      round(float(score), 4),
        })
    return results


def build_prompt(question: str, chunks: list[dict]) -> str:
    """Build the prompt that gets sent to Ollama."""
    context = ""
    for i, chunk in enumerate(chunks, 1):
        context += f"\n[Source {i}: {chunk['title']}]\n{chunk['chunk_text']}\n"

    prompt = f"""You are a helpful assistant that answers questions based on Nepali news articles.
Use the context below to answer.
Answer in the same language as the question.

Context:
{context}

Question: {question}

Answer:"""
    return prompt


def ask_ollama(prompt: str) -> str:
    """Send prompt to Ollama and return the response text."""
    payload = {
        "model":  OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False,        # get full response at once
    }
    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["response"].strip()
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=503,
            detail="Ollama is not running. Start it with: ollama serve"
        )
    except requests.exceptions.Timeout:
        raise HTTPException(
            status_code=504,
            detail="Ollama took too long to respond. Try a smaller model."
        )



@app.get("/")
def root():
    return {"status": "running", "model": OLLAMA_MODEL}


@app.get("/health")
def health():
    return {
        "status":       "ok",
        "chunks":       len(chunk_map),
        "index_vectors": index.ntotal,
        "embed_model":  MODEL_NAME,
        "llm":          OLLAMA_MODEL,
    }


@app.post("/ask", response_model=AskResponse)
def ask(request: AskRequest):
    question = request.question.strip()

    if not question:
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    chunks = retrieve(question)

    if not chunks:
        raise HTTPException(status_code=404, detail="No relevant chunks found.")

    prompt = build_prompt(question, chunks)

    answer = ask_ollama(prompt)

    return AskResponse(
        question=question,
        answer=answer,
        sources=[Source(**c) for c in chunks],
    )