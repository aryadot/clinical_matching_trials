"""
pipeline/embeddings.py — Vector Embeddings & ChromaDB Operations
Embeds patient profiles and trial descriptions using sentence-transformers,
stores them in ChromaDB, and performs semantic retrieval.
"""

import streamlit as st
import chromadb
import os
import tempfile
from config import EMBEDDING_MODEL, CHROMA_COLLECTION


@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedding_model():
    """Load sentence-transformers model for vector embeddings. Runs locally."""
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBEDDING_MODEL)


@st.cache_resource(show_spinner="Initializing vector store...")
def get_chroma_client():
    """Initialize ChromaDB persistent client."""
    persist_dir = os.path.join(tempfile.gettempdir(), "clinical_trials_chroma")
    return chromadb.PersistentClient(path=persist_dir)


def get_collection(client):
    """Get or create the trials collection."""
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )


def embed_text(text: str) -> list[float]:
    """Generate embedding vector for a single text."""
    model = load_embedding_model()
    return model.encode(text).tolist()


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Generate embedding vectors for a batch of texts."""
    model = load_embedding_model()
    return model.encode(texts).tolist()


def index_trials(trials: list[dict], trial_texts: list[str]):
    """
    Index all trial embedding texts into ChromaDB.
    Skips if already indexed (idempotent).
    """
    client = get_chroma_client()
    collection = get_collection(client)

    # Check if already indexed
    existing = collection.count()
    if existing >= len(trials):
        return collection

    # Clear and re-index
    client.delete_collection(CHROMA_COLLECTION)
    collection = client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )

    # Batch embed and insert
    model = load_embedding_model()
    batch_size = 64

    for i in range(0, len(trials), batch_size):
        batch_texts = trial_texts[i:i + batch_size]
        batch_trials = trials[i:i + batch_size]
        batch_embeddings = model.encode(batch_texts).tolist()

        ids = [t["trial_id"] for t in batch_trials]
        metadatas = [{
            "title": t.get("title", "")[:500],
            "conditions": t.get("conditions", "")[:500],
            "interventions": t.get("interventions", "")[:500],
            "status": t.get("status", ""),
        } for t in batch_trials]

        collection.add(
            ids=ids,
            embeddings=batch_embeddings,
            documents=batch_texts,
            metadatas=metadatas,
        )

    return collection


def semantic_search(query_text: str, collection, top_k: int = 20) -> list[dict]:
    """
    Search ChromaDB for trials most semantically similar to the query.
    Returns list of {trial_id, score, title, conditions, interventions, text}.
    """
    query_embedding = embed_text(query_text)

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["documents", "metadatas", "distances"],
    )

    matches = []
    for i in range(len(results["ids"][0])):
        # ChromaDB returns cosine distance; convert to similarity
        distance = results["distances"][0][i]
        similarity = 1.0 - distance  # cosine similarity

        matches.append({
            "trial_id": results["ids"][0][i],
            "similarity": round(similarity, 4),
            "text": results["documents"][0][i],
            "title": results["metadatas"][0][i].get("title", ""),
            "conditions": results["metadatas"][0][i].get("conditions", ""),
            "interventions": results["metadatas"][0][i].get("interventions", ""),
            "status": results["metadatas"][0][i].get("status", ""),
        })

    return matches
