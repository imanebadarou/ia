#!/usr/bin/env python3
"""
Simplified RAG System for Dragon Ball Wiki
Using basic components without heavy LangChain dependencies
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import List, Dict, Any
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from dotenv import load_dotenv
import faiss

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class SimpleDragonBallRAG:
    def __init__(self, corpus_dir: str = "corpus/saga_freezer"):
        self.corpus_dir = Path(corpus_dir)
        self.model = None
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
                if content.strip():  # Skip empty files
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
        """Build the FAISS index"""
        print("🏗️  Construction de l'index RAG...")

        # Load corpus
        documents = self.load_corpus()

        # Initialize sentence transformer
        print("🧮 Chargement du modèle d'embeddings...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')

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
        embeddings = self.model.encode(self.chunks, show_progress_bar=True)

        # Create FAISS index
        print("💾 Création de l'index FAISS...")
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product (cosine similarity)

        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)
        self.index.add(embeddings)

        print("✅ Index RAG construit avec succès!")

    def save_index(self, path: str = "simple_vectorstore"):
        """Save the index and metadata"""
        if self.index and self.model:
            # Save FAISS index
            faiss.write_index(self.index, f"{path}.faiss")

            # Save metadata
            metadata = {
                'chunks': self.chunks,
                'metadata': self.chunk_metadata
            }
            with open(f"{path}_metadata.json", 'w', encoding='utf-8') as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            print(f"💾 Index sauvegardé dans: {path}")

    def load_index(self, path: str = "simple_vectorstore"):
        """Load the index and metadata"""
        try:
            # Load FAISS index
            self.index = faiss.read_index(f"{path}.faiss")

            # Load metadata
            with open(f"{path}_metadata.json", 'r', encoding='utf-8') as f:
                metadata = json.load(f)

            self.chunks = metadata['chunks']
            self.chunk_metadata = metadata['metadata']

            # Load model
            self.model = SentenceTransformer('all-MiniLM-L6-v2')

            print(f"📂 Index chargé depuis: {path}")
            return True
        except FileNotFoundError:
            return False

    def search(self, query: str, k: int = 5) -> List[Dict[str, Any]]:
        """Search for relevant chunks"""
        if not self.index or not self.model:
            raise ValueError("Index non initialisé. Appelez build_index() d'abord.")

        print(f"🔍 Recherche pour: '{query}'")

        # Encode query
        query_embedding = self.model.encode([query])
        faiss.normalize_L2(query_embedding)

        # Search
        scores, indices = self.index.search(query_embedding, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < len(self.chunks):  # Valid index
                results.append({
                    'chunk': self.chunks[idx],
                    'metadata': self.chunk_metadata[idx],
                    'score': float(score)
                })

        print(f"📋 {len(results)} chunks pertinents trouvés")
        return results

    def generate_response(self, query: str, search_results: List[Dict[str, Any]], model: str = "meta-llama/llama-3.2-3b-instruct:free") -> str:
        """Generate response using OpenRouter"""
        if not self.client:
            return "❌ Clé API OpenRouter manquante"

        print("🤖 Génération de la réponse...")

        # Build context from top chunks
        context_parts = []
        for i, result in enumerate(search_results[:3], 1):  # Use top 3 chunks
            chunk_preview = result['chunk'][:500] + "..." if len(result['chunk']) > 500 else result['chunk']
            context_parts.append(f"Source {i}: {chunk_preview}")

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

    def ask_question(self, query: str, k: int = 5, model: str = "meta-llama/llama-3.2-3b-instruct:free") -> Dict[str, Any]:
        """Main method to ask a question"""
        # Search for relevant chunks
        search_results = self.search(query, k)

        # Generate response
        answer = self.generate_response(query, search_results, model)

        return {
            "question": query,
            "answer": answer,
            "num_results": len(search_results)
        }


def main():
    # Initialize RAG system
    rag = SimpleDragonBallRAG()

    # Try to load existing index
    if not rag.load_index():
        print("🔨 Construction de l'index pour la première fois...")
        rag.build_index()
        rag.save_index()

    # Interactive interface
    print("\n" + "="*50)
    print("🤖 RAG System - Dragon Ball Expert (Simplified)")
    print("="*50)
    print("Posez vos questions sur Dragon Ball!")
    print("Tapez 'quit' pour quitter")
    print()

    while True:
        try:
            query = input("❓ Votre question: ").strip()
            if query.lower() in ['quit', 'exit', 'q']:
                break

            if query:
                result = rag.ask_question(query)
                print(f"\n🤖 Réponse: {result['answer']}")
                print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

    print("\n👋 Au revoir!")


if __name__ == "__main__":
    main()