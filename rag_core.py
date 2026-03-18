#!/usr/bin/env python3
"""Core RAG logic for the Dragon Ball Streamlit interface.

This module contains the retrieval+generation logic used by the web UI.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Any

import faiss
from dotenv import load_dotenv
import openai
from openai import OpenAI
from sentence_transformers import SentenceTransformer, CrossEncoder

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_MODEL = os.getenv("OPENROUTER_MODEL")

# Default paths
DEFAULT_CORPUS_DIR = "corpus/saga_freezer"
DEFAULT_VECTORSTORE = "vectorstore"
DEFAULT_ADVANCED_VECTORSTORE = "advanced_vectorstore"


class DragonBallRAG:
    """Retrieval + generation RAG system."""

    def __init__(
        self,
        corpus_dir: str = DEFAULT_CORPUS_DIR,
        vectorstore_path: str = DEFAULT_VECTORSTORE,
        use_reranking: bool = False,
        rerank_model: str = "cross-encoder/ms-marco-MiniLM-L-6-v2",
    ):
        self.corpus_dir = Path(corpus_dir)
        self.vectorstore_path = vectorstore_path
        self.use_reranking = use_reranking

        self.model = None
        self.reranker = None
        self.index = None
        self.chunks: List[str] = []
        self.chunk_metadata: List[Dict[str, Any]] = []

        self.client = None
        if OPENROUTER_API_KEY:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )

        if self.use_reranking:
            self.rerank_model = rerank_model

    def _load_corpus(self) -> List[str]:
        """Load text files from the corpus directory."""
        docs = []
        for file_path in sorted(self.corpus_dir.glob("*.txt")):
            try:
                text = file_path.read_text(encoding="utf-8").strip()
                if text:
                    docs.append(text)
            except Exception:
                pass
        return docs

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = max(end - overlap, end)
        return chunks

    def build_index(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """Build the FAISS index from the corpus."""
        docs = self._load_corpus()

        self.model = SentenceTransformer("all-MiniLM-L6-v2")
        if self.use_reranking:
            self.reranker = CrossEncoder(self.rerank_model)

        self.chunks = []
        self.chunk_metadata = []

        for doc_id, doc in enumerate(docs):
            for chunk_id, chunk in enumerate(self._chunk_text(doc, chunk_size, chunk_overlap)):
                self.chunks.append(chunk)
                self.chunk_metadata.append({
                    "doc_id": doc_id,
                    "chunk_id": chunk_id,
                })

        embeddings = self.model.encode(self.chunks, show_progress_bar=False)
        faiss.normalize_L2(embeddings)

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dim)
        self.index.add(embeddings)

        self._save_index()

    def _save_index(self):
        if self.index is None:
            return
        faiss.write_index(self.index, f"{self.vectorstore_path}.faiss")
        metadata = {"chunks": self.chunks, "metadata": self.chunk_metadata}
        Path(f"{self.vectorstore_path}_meta.json").write_text(
            __import__("json").dumps(metadata, ensure_ascii=False), encoding="utf-8"
        )

    def load_index(self) -> bool:
        try:
            self.index = faiss.read_index(f"{self.vectorstore_path}.faiss")
            meta = __import__("json").loads(
                Path(f"{self.vectorstore_path}_meta.json").read_text(encoding="utf-8")
            )
            self.chunks = meta["chunks"]
            self.chunk_metadata = meta["metadata"]
            self.model = SentenceTransformer("all-MiniLM-L6-v2")
            if self.use_reranking:
                self.reranker = CrossEncoder(self.rerank_model)
            return True
        except Exception:
            return False

    def _vector_search(self, query: str, k: int = 5):
        if self.index is None or self.model is None:
            raise ValueError("Index not loaded")
        q_emb = self.model.encode([query])
        faiss.normalize_L2(q_emb)
        scores, idxs = self.index.search(q_emb, k)
        results = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx < len(self.chunks):
                results.append({
                    "chunk": self.chunks[idx],
                    "score": float(score),
                    "metadata": self.chunk_metadata[idx],
                })
        return results

    def _rerank(self, query: str, candidates: List[Dict[str, Any]], top_k: int = 5):
        if self.reranker is None:
            return candidates[:top_k]

        pairs = [[query, c["chunk"]] for c in candidates]
        rerank_scores = self.reranker.predict(pairs)

        for c, s in zip(candidates, rerank_scores):
            c["rerank_score"] = float(s)
            c["combined_score"] = 0.7 * c["rerank_score"] + 0.3 * c["score"]

        candidates.sort(key=lambda x: x["combined_score"], reverse=True)
        return candidates[:top_k]

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        candidates = self._vector_search(query, k * 4)
        if self.use_reranking:
            return self._rerank(query, candidates, top_k=k)
        return candidates[:k]

    def generate(self, query: str, context_chunks: List[Dict[str, Any]], model: str = "meta-llama/llama-3.2-3b-instruct:free") -> str:
        if not self.client:
            return "❌ OPENROUTER_API_KEY missing (définissez OPENROUTER_API_KEY dans .env)"

        # Autoriser le modèle à être surchargé via une variable d'environnement.
        model_to_use = OPENROUTER_MODEL or model
        max_retries = int(os.getenv("OPENROUTER_RETRIES", "3"))

        context = "\n\n".join(
            f"Source {i+1}: {chunk['chunk']}" for i, chunk in enumerate(context_chunks)
        )

        prompt = f"""Tu es un expert Dragon Ball. Réponds précisément à la question en utilisant seulement les informations suivantes.

Question: {query}

Contexte:
{context}

Réponse:"""

        backoff = 1.0
        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.completions.create(
                    model=model_to_use,
                    messages=[
                        {"role": "system", "content": "Tu es un expert Dragon Ball. Réponds de manière précise et concise."},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=800,
                    temperature=0.3,
                )

                return response.choices[0].message.content.strip()

            except openai.RateLimitError:
                if attempt == max_retries:
                    return (
                        "❌ Taux de requêtes dépassé (429). Réessayez dans quelques instants, "
                        "ou utilisez votre propre clé OpenRouter (OPENROUTER_API_KEY dans .env)"
                    )
                time.sleep(backoff)
                backoff *= 2

            except openai.OpenAIError as e:
                return (
                    f"❌ Erreur OpenRouter ({type(e).__name__}): {e}. "
                    "Vérifiez votre clé et votre modèle (OPENROUTER_API_KEY / OPENROUTER_MODEL)."
                )

            except Exception as e:
                return f"❌ Erreur inattendue lors de la génération : {e}"
