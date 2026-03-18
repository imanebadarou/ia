#!/usr/bin/env python3
"""
RAG System avec Reranking avancé
Utilise un cross-encoder pour améliorer la pertinence des résultats
"""

import os
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer, CrossEncoder
from openai import OpenAI
from dotenv import load_dotenv
import faiss
import numpy as np

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class AdvancedDragonBallRAG:
    def __init__(self, corpus_dir: str = "corpus/saga_freezer"):
        self.corpus_dir = Path(corpus_dir)
        self.embedding_model = None
        self.reranker = None
        self.index = None
        self.chunks = []
        self.chunk_metadata = []
        self.client = None

        # Initialize OpenRouter client
        if OPENROUTER_API_KEY:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )

    def load_corpus(self) -> List[str]:
        """Load all text files from corpus"""
        print("📚 Chargement du corpus...")
        documents = []
        for file_path in self.corpus_dir.glob("*.txt"):
            try:
                content = file_path.read_text(encoding='utf-8')
                if content.strip():
                    documents.append(content)
                    print(f"  ✅ {file_path.name} ({len(content)} caractères)")
            except Exception as e:
                print(f"  ❌ Erreur avec {file_path.name}: {e}")
        print(f"📊 Corpus chargé: {len(documents)} documents")
        return documents

    def chunk_text(self, text: str, chunk_size: int = 1000, overlap: int = 200) -> List[str]:
        """Split text into chunks"""
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks

    def build_index(self):
        """Build the FAISS index with reranker"""
        print("🏗️  Construction de l'index RAG avancé...")

        # Load corpus
        documents = self.load_corpus()

        # Initialize models
        print("🧮 Chargement des modèles...")
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')

        # Chunk documents
        print("✂️  Découpage en chunks...")
        all_chunks = []
        all_metadata = []

        for doc_idx, doc in enumerate(documents):
            chunks = self.chunk_text(doc)
            for chunk_idx, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadata.append({
                    'doc_id': doc_idx,
                    'chunk_id': chunk_idx,
                    'length': len(chunk)
                })

        self.chunks = all_chunks
        self.chunk_metadata = all_metadata
        print(f"📊 Total chunks: {len(self.chunks)}")

        # Create embeddings
        print("🧮 Création des embeddings...")
        embeddings = self.embedding_model.encode(self.chunks, show_progress_bar=True)

        # Create FAISS index
        print("💾 Création de l'index FAISS...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        print("✅ Index RAG avancé construit avec succès!")

    def save_index(self, path: str = "advanced_vectorstore"):
        """Save the index and metadata"""
        if self.index and self.embedding_model:
            faiss.write_index(self.index, f"{path}.faiss")
            metadata = {
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }
            import json
            with open(f"{path}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)
            print(f"💾 Index avancé sauvegardé dans: {path}")

    def load_index(self, path: str = "advanced_vectorstore"):
        """Load the index and metadata"""
        try:
            self.index = faiss.read_index(f"{path}.faiss")
            import json
            with open(f"{path}_metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            self.chunks = metadata['chunks']
            self.chunk_metadata = metadata['metadata']
            self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
            self.reranker = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
            print(f"📂 Index avancé chargé depuis: {path}")
            return True
        except FileNotFoundError:
            return False

    def search_with_reranking(self, query: str, initial_k: int = 20, final_k: int = 5) -> List[Dict[str, Any]]:
        """Recherche avec reranking en deux étapes"""
        if not self.index or not self.embedding_model or not self.reranker:
            raise ValueError("Index non initialisé. Appelez build_index() d'abord.")

        print(f"🔍 Recherche avancée pour: '{query}'")

        # Étape 1: Recherche vectorielle rapide (récupère plus de candidats)
        print(f"   📊 Étape 1: Recherche vectorielle (top {initial_k})...")
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)

        scores, indices = self.index.search(query_embedding, initial_k)

        # Préparer les candidats pour le reranking
        candidates = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):
                candidates.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'initial_score': float(score),
                    'index': idx
                })

        # Étape 2: Reranking avec cross-encoder
        print(f"   🎯 Étape 2: Reranking (top {final_k})...")

        # Préparer les paires (question, chunk) pour le reranker
        pairs = [[query, candidate['chunk']] for candidate in candidates]

        # Obtenir les scores de reranking
        rerank_scores = self.reranker.predict(pairs)

        # Combiner avec les scores initiaux et trier
        for i, candidate in enumerate(candidates):
            candidate['rerank_score'] = float(rerank_scores[i])
            candidate['combined_score'] = 0.7 * candidate['rerank_score'] + 0.3 * candidate['initial_score']

        # Trier par score combiné et prendre les top final_k
        candidates.sort(key=lambda x: x['combined_score'], reverse=True)
        top_results = candidates[:final_k]

        print(f"📋 {len(top_results)} chunks finaux sélectionnés:")
        for i, result in enumerate(top_results, 1):
            print(f"  {i}. Score initial: {result['initial_score']:.4f}")
            print(f"     Score rerank: {result['rerank_score']:.4f}")
            print(f"     Score combiné: {result['combined_score']:.4f}")
            preview = result['chunk'][:100].replace('\n', ' ')
            print(f"     {preview}...")

        return top_results

    def generate_response(self, query: str, search_results: List[Dict[str, Any]], model: str = "meta-llama/llama-3.2-3b-instruct:free") -> str:
        """Generate response using OpenRouter"""
        if not self.client:
            return "❌ Clé API OpenRouter manquante"

        print("🤖 Génération de la réponse...")

        # Build context from reranked chunks
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):  # Use top 3 reranked chunks
            chunk_preview = result['chunk'][:500] + "..." if len(result['chunk']) > 500 else result['chunk']
            context_parts.append(f"Source {i} (score: {result['combined_score']:.3f}):\n{chunk_preview}")

        context = "\n\n".join(context_parts)

        # Build prompt
        prompt = f"""Tu es un expert de Dragon Ball. Réponds à la question de manière précise et concise en utilisant uniquement les informations fournies.

Question: {query}

Informations disponibles:
{context}

Réponse:"""

        try:
            response = self.client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "Tu es un expert de Dragon Ball. Réponds de manière précise et basée uniquement sur les informations fournies."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1000,
                temperature=0.3
            )

            answer = response.choices[0].message.content.strip()
            print("✅ Réponse générée!")
            return answer

        except Exception as e:
            return f"❌ Erreur lors de la génération: {e}"

    def ask_question_advanced(self, query: str, initial_k: int = 20, final_k: int = 5, model: str = "meta-llama/llama-3.2-3b-instruct:free") -> Dict[str, Any]:
        """Main method with advanced reranking"""
        # Search with reranking
        search_results = self.search_with_reranking(query, initial_k, final_k)

        # Generate response
        answer = self.generate_response(query, search_results, model)

        return {
            "question": query,
            "answer": answer,
            "num_results": len(search_results),
            "reranked": True
        }


def compare_systems():
    """Compare basic RAG vs Advanced RAG with reranking"""
    print("🔬 Comparaison: RAG Basique vs RAG Avancé")
    print("=" * 60)

    # Test question
    question = "Quelle est la puissance de Frieza ?"

    # Basic RAG
    print("🐌 RAG BASIQUE:")
    from simple_rag import SimpleDragonBallRAG
    basic_rag = SimpleDragonBallRAG()
    if basic_rag.load_index("simple_vectorstore"):
        basic_results = basic_rag.search(question, k=5)
        print(f"   📊 {len(basic_results)} résultats")
        for i, result in enumerate(basic_results[:3], 1):
            print(f"     {i}. Score: {result.get('score', 'N/A'):.3f}")
    else:
        print("   ❌ Index basique non trouvé")

    # Advanced RAG
    print("\n🚀 RAG AVANCÉ (avec reranking):")
    advanced_rag = AdvancedDragonBallRAG()
    if advanced_rag.load_index():
        advanced_results = advanced_rag.search_with_reranking(question, initial_k=10, final_k=5)
        print(f"   📊 {len(advanced_results)} résultats rerankés")
        for i, result in enumerate(advanced_results[:3], 1):
            print(f"     {i}. Score combiné: {result['combined_score']:.3f}")
            print(f"        Score initial: {result['initial_score']:.3f}")
            print(f"        Score rerank: {result['rerank_score']:.3f}")
    else:
        print("   ❌ Index avancé non trouvé")


def main():
    # Initialize advanced RAG system
    rag = AdvancedDragonBallRAG()

    # Try to load existing index
    if not rag.load_index():
        print("🔨 Construction de l'index avancé pour la première fois...")
        rag.build_index()
        rag.save_index()

    # Interactive interface
    print("\n" + "="*60)
    print("🚀 RAG System Avancé - Dragon Ball Expert (avec Reranking)")
    print("="*60)
    print("Posez vos questions sur Dragon Ball!")
    print("Le système utilise un reranking avancé pour de meilleurs résultats")
    print("Tapez 'quit' pour quitter, 'compare' pour comparer les systèmes")
    print()

    while True:
        try:
            query = input("❓ Votre question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break
            elif query.lower() == 'compare':
                compare_systems()
                continue

            if query:
                result = rag.ask_question_advanced(query)
                print(f"\n🤖 Réponse: {result['answer']}")
                print("-" * 60)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

    print("\n👋 Au revoir!")


if __name__ == "__main__":
    main()