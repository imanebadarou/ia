#!/usr/bin/env python3
"""
RAG System for Dragon Ball Wiki
Retrieval-Augmented Generation using LangChain, FAISS, and OpenRouter
"""

import os
import json
from pathlib import Path
from typing import List, Dict, Any

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from openai import OpenAI
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")

class DragonBallRAG:
    def __init__(self, corpus_dir: str = "corpus/saga_freezer"):
        self.corpus_dir = Path(corpus_dir)
        self.vectorstore = None
        self.embedding_model = None
        self.client = None

        # Initialize OpenRouter client
        if OPENROUTER_API_KEY:
            self.client = OpenAI(
                base_url="https://openrouter.ai/api/v1",
                api_key=OPENROUTER_API_KEY,
            )
        else:
            print("⚠️  OPENROUTER_API_KEY not found in .env file")

    def load_corpus(self) -> List[Document]:
        """I) Ingestion — Charger le corpus"""
        print("📚 Chargement du corpus...")

        documents = []
        for file_path in self.corpus_dir.glob("*.txt"):
            try:
                content = file_path.read_text(encoding='utf-8')
                doc = Document(
                    page_content=content,
                    metadata={"source": file_path.name, "title": file_path.stem}
                )
                documents.append(doc)
                print(f"  ✅ {file_path.name} ({len(content)} caractères)")
            except Exception as e:
                print(f"  ❌ Erreur avec {file_path.name}: {e}")

        print(f"📊 Corpus chargé: {len(documents)} documents")
        return documents

    def chunk_documents(self, documents: List[Document], chunk_size: int = 1000, chunk_overlap: int = 200) -> List[Document]:
        """II) Découper le corpus en chunks"""
        print("✂️  Découpage en chunks...")

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        chunks = []
        for doc in documents:
            doc_chunks = text_splitter.split_documents([doc])
            chunks.extend(doc_chunks)
            print(f"  📄 {doc.metadata['title']}: {len(doc_chunks)} chunks")

        print(f"📊 Total chunks: {len(chunks)}")
        return chunks

    def create_embeddings(self, chunks: List[Document]) -> FAISS:
        """III) Vectoriser les chunks et IV) Stocker dans FAISS"""
        print("🧮 Création des embeddings...")

        # Utiliser le modèle recommandé
        self.embedding_model = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2",
            model_kwargs={'device': 'cpu'}  # Utiliser CPU
        )

        print("💾 Stockage dans FAISS...")
        self.vectorstore = FAISS.from_documents(chunks, self.embedding_model)

        print("✅ Base de données vectorielle créée!")
        return self.vectorstore

    def save_vectorstore(self, path: str = "vectorstore"):
        """Sauvegarder la base vectorielle"""
        if self.vectorstore:
            self.vectorstore.save_local(path)
            print(f"💾 Base sauvegardée dans: {path}")

    def load_vectorstore(self, path: str = "vectorstore"):
        """Charger la base vectorielle sauvegardée"""
        if os.path.exists(path):
            self.vectorstore = FAISS.load_local(path, self.embedding_model)
            print(f"📂 Base chargée depuis: {path}")
            return True
        return False

    def retrieve_relevant_chunks(self, query: str, k: int = 5) -> List[Document]:
        """V) Retrieval — Recherche d'information"""
        if not self.vectorstore:
            raise ValueError("Base vectorielle non initialisée. Appelez create_embeddings() d'abord.")

        print(f"🔍 Recherche pour: '{query}'")

        # Vectoriser la question
        relevant_docs = self.vectorstore.similarity_search(query, k=k)

        print(f"📋 {len(relevant_docs)} chunks pertinents trouvés:")
        for i, doc in enumerate(relevant_docs, 1):
            source = doc.metadata.get('title', 'Unknown')
            preview = doc.page_content[:100].replace('\n', ' ')
            print(f"  {i}. {source}: {preview}...")

        return relevant_docs

    def generate_response(self, query: str, relevant_chunks: List[Document], model: str = "meta-llama/llama-3.2-3b-instruct:free") -> str:
        """VI) Génération de la réponse"""
        if not self.client:
            return "❌ Clé API OpenRouter manquante"

        print("🤖 Génération de la réponse...")

        # Construire le contexte
        context = "\n\n".join([
            f"Source: {doc.metadata.get('title', 'Unknown')}\n{doc.page_content}"
            for doc in relevant_chunks
        ])

        # Construire le prompt
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
        """Méthode principale pour poser une question"""
        # Récupérer les chunks pertinents
        relevant_chunks = self.retrieve_relevant_chunks(query, k)

        # Générer la réponse
        answer = self.generate_response(query, relevant_chunks, model)

        return {
            "question": query,
            "answer": answer,
            "sources": [doc.metadata.get('title', 'Unknown') for doc in relevant_chunks],
            "num_chunks": len(relevant_chunks)
        }

    def build_index(self, save_path: str = "vectorstore"):
        """Construction complète de l'index"""
        print("🏗️  Construction de l'index RAG...")

        # Étape 1: Charger le corpus
        documents = self.load_corpus()

        # Étape 2: Chunking
        chunks = self.chunk_documents(documents)

        # Étape 3-4: Embeddings et stockage
        self.create_embeddings(chunks)

        # Sauvegarder
        self.save_vectorstore(save_path)

        print("🎉 Index RAG construit avec succès!")


def main():
    # Initialiser le système RAG
    rag = DragonBallRAG(corpus_dir="corpus/saga_freezer")

    # Vérifier si l'index existe déjà
    if not rag.load_vectorstore():
        print("🔨 Construction de l'index pour la première fois...")
        rag.build_index()
    else:
        print("📂 Index existant chargé!")

    # Interface interactive
    print("\n" + "="*50)
    print("🤖 RAG System - Dragon Ball Expert")
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
                print(f"📚 Sources: {', '.join(result['sources'])}")
                print("-" * 50)

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"❌ Erreur: {e}")

    print("\n👋 Au revoir!")


if __name__ == "__main__":
    main()