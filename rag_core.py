"""
rag_core.py — Shared RAG logic (FAISS backend, no sqlite3 required)
"""

import json
import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer

_embedding_model: Optional[SentenceTransformer] = None


def _get_embedding_model(model_name: str) -> SentenceTransformer:
    global _embedding_model
    if _embedding_model is None:
        print(f"[rag_core] Loading embedding model: {model_name}", file=sys.stderr)
        _embedding_model = SentenceTransformer(model_name)
    return _embedding_model


class VectorStore:
    """FAISS + JSON metadata store. No sqlite3 dependency."""

    def __init__(self, persist_dir: str):
        import faiss
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index_path = self.persist_dir / "index.faiss"
        self.meta_path = self.persist_dir / "metadata.json"
        self.metadata: list[dict] = []
        self.id_set: set[str] = set()
        self._index = None
        self._load()

    def _load(self):
        import faiss
        if self.index_path.exists() and self.meta_path.exists():
            self._index = faiss.read_index(str(self.index_path))
            with open(self.meta_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            self.metadata = data.get("metadata", [])
            self.id_set = {m["id"] for m in self.metadata}

    def _save(self):
        import faiss
        if self._index is not None:
            faiss.write_index(self._index, str(self.index_path))
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump({"metadata": self.metadata}, f, ensure_ascii=False, indent=2)

    def count(self) -> int:
        return len(self.metadata)

    def clear(self):
        self._index = None
        self.metadata = []
        self.id_set = set()
        for p in [self.index_path, self.meta_path]:
            if p.exists():
                p.unlink()

    def upsert(self, ids: list[str], documents: list[str],
               embeddings: list[list[float]], metadatas: list[dict]):
        import faiss
        if not ids:
            return
        # Only add truly new IDs (skip duplicates for simplicity)
        new_indices = [i for i, cid in enumerate(ids) if cid not in self.id_set]
        if not new_indices:
            return
        vecs = np.array([embeddings[i] for i in new_indices], dtype=np.float32)
        faiss.normalize_L2(vecs)
        dim = vecs.shape[1]
        if self._index is None:
            self._index = faiss.IndexFlatIP(dim)
        self._index.add(vecs)
        for i in new_indices:
            meta = {"id": ids[i], "text": documents[i], **metadatas[i]}
            self.metadata.append(meta)
            self.id_set.add(ids[i])
        self._save()

    def query(self, query_embedding: list[float], top_k: int) -> list[dict]:
        import faiss
        if self._index is None or self.count() == 0:
            return []
        vec = np.array([query_embedding], dtype=np.float32)
        faiss.normalize_L2(vec)
        k = min(top_k, self.count())
        scores, indices = self._index.search(vec, k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if 0 <= idx < len(self.metadata):
                m = self.metadata[idx]
                results.append({
                    "text": m.get("text", ""),
                    "source": m.get("source", "unknown"),
                    "chunk_index": m.get("chunk_index", 0),
                    "score": float(score),
                })
        return results


def get_vector_store(persist_dir: str) -> VectorStore:
    return VectorStore(persist_dir)


def embed_texts(texts: list[str],
                model_name: str = "paraphrase-multilingual-MiniLM-L12-v2") -> list[list[float]]:
    if not texts:
        return []
    model = _get_embedding_model(model_name)
    return model.encode(texts, show_progress_bar=False).tolist()


def retrieve(query: str, store: VectorStore, top_k: int = 5,
             threshold: float = 0.0,
             embedding_model: str = "paraphrase-multilingual-MiniLM-L12-v2") -> list[dict]:
    if store.count() == 0:
        return []
    qvec = embed_texts([query], model_name=embedding_model)[0]
    results = store.query(qvec, top_k=top_k)
    return [
        {**r, "distance": 1.0 - r["score"], "similarity": r["score"]}
        for r in results if r["score"] >= threshold
    ]


def generate_answer(query: str, chunks: list[dict], history: list[dict],
                    model: str = "gemini/gemini-2.5-flash") -> str:
    if not chunks:
        return ("No relevant information found in the knowledge base. "
                "Try rephrasing or run `data_update.py --rebuild`.")
    context = "\n\n---\n\n".join(
        f"[{i}] Source: {c['source']} (chunk {c['chunk_index']})\n{c['text']}"
        for i, c in enumerate(chunks, 1)
    )
    system_prompt = (
        "You are a knowledgeable AI research assistant. "
        "Answer based ONLY on the provided context. "
        "Cite sources using [N] notation. "
        "If context is insufficient, say so clearly. "
        "Always respond in the same language as the user's question."
    )
    messages = [{"role": "system", "content": system_prompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": f"Context:\n\n{context}\n\nQuestion: {query}"})

    api_base = os.environ.get("LITELLM_BASE_URL", "").strip()
    api_key = os.environ.get("LITELLM_API_KEY", "").strip()

    # Try litellm first; fall back to direct HTTP (Ollama) if litellm is broken
    try:
        from litellm import completion
        kwargs: dict = {"model": model, "messages": messages}
        if api_base and "localhost" not in api_base and "127.0.0.1" not in api_base:
            kwargs["api_base"] = api_base
        if api_key and api_key != "ollama":
            kwargs["api_key"] = api_key
        response = completion(**kwargs)
        return response.choices[0].message.content
    except (ImportError, Exception) as litellm_err:
        # Fall back to direct Ollama HTTP API
        import requests as _requests
        # Determine Ollama base URL
        ollama_base = api_base if api_base else "http://localhost:11434"
        # Extract model name: strip provider prefix if present
        ollama_model = model
        if "/" in model:
            parts = model.split("/")
            if parts[0].lower() in ("gemini", "openai", "anthropic", "cohere", "ollama"):
                ollama_model = "/".join(parts[1:])
        # Check available models and pick the first one if the requested model isn't available
        try:
            tags_resp = _requests.get(f"{ollama_base}/api/tags", timeout=5)
            available = [m["name"] for m in tags_resp.json().get("models", [])]
            if available and ollama_model not in available:
                # Try matching by base name (without tag)
                base_name = ollama_model.split(":")[0]
                matched = [m for m in available if m.startswith(base_name)]
                ollama_model = matched[0] if matched else available[0]
        except Exception:
            pass
        url = f"{ollama_base}/api/chat"
        payload = {
            "model": ollama_model,
            "messages": messages,
            "stream": False,
        }
        resp = _requests.post(url, json=payload, timeout=90)
        resp.raise_for_status()
        data = resp.json()
        return data["message"]["content"]


def format_citations(chunks: list[dict]) -> str:
    if not chunks:
        return ""
    lines = ["\n📚 Sources:"]
    for i, c in enumerate(chunks, 1):
        lines.append(f"  [{i}] {c.get('source','?')} "
                     f"(chunk {c.get('chunk_index',0)}, "
                     f"similarity: {c.get('similarity',0):.3f})")
    return "\n".join(lines)


def check_required_env_vars(required: list[str]) -> None:
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        for var in missing:
            print(f"[ERROR] Missing env var '{var}'. Copy .env.example to .env.",
                  file=sys.stderr)
        sys.exit(1)
