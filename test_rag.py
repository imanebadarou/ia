#!/usr/bin/env python3
"""
Test script for the Dragon Ball RAG system
"""

import os
from rag_system import DragonBallRAG
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

def test_rag_system():
    """Test the RAG system with some questions"""

    # Initialize RAG system
    rag = DragonBallRAG()

    # Load existing vectorstore
    if not rag.load_vectorstore():
        print("❌ Vectorstore not found. Run rag_system.py first to build the index.")
        return

    # Test questions
    test_questions = [
        "Qui est Frieza ?",
        "Qu'est-ce qu'un Super Saiyan ?",
        "Où se déroule la Frieza Saga ?",
        "Qui sont les membres de la Ginyu Force ?",
        "Quelles sont les Dragon Balls ?"
    ]

    print("🧪 Test du système RAG - Dragon Ball")
    print("=" * 50)

    for question in test_questions:
        print(f"\n❓ Question: {question}")
        try:
            result = rag.ask_question(question, k=3)
            print(f"🤖 Réponse: {result['answer']}")
            print(f"📚 Sources: {', '.join(result['sources'])}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_rag_system()