#!/usr/bin/env python3
"""
Demo of the RAG system without API calls - shows retrieval only
"""

from simple_rag import SimpleDragonBallRAG

def demo_rag_retrieval():
    """Demo the RAG retrieval system without API calls"""

    # Initialize RAG system
    rag = SimpleDragonBallRAG()

    # Load existing index
    if not rag.load_index():
        print("❌ Index non trouvé. Lancez d'abord test_simple_rag.py")
        return

    # Test questions with retrieval only
    test_questions = [
        "Qui est Frieza ?",
        "Qu'est-ce qu'un Super Saiyan ?",
        "Où se déroule la Frieza Saga ?",
        "Qui sont les membres de la Ginyu Force ?",
        "Quelles sont les Dragon Balls ?"
    ]

    print("🎯 Démo du système RAG - Recherche seulement")
    print("=" * 60)
    print("Le système trouve les chunks pertinents dans votre corpus Dragon Ball")
    print()

    for question in test_questions:
        print(f"❓ Question: {question}")

        # Get relevant chunks
        results = rag.search(question, k=2)

        print("📋 Chunks pertinents trouvés:")
        for i, result in enumerate(results, 1):
            # Show preview of the chunk
            preview = result['chunk'][:300] + "..." if len(result['chunk']) > 300 else result['chunk']
            score = result['score']
            print(f"  {i}. Score: {score:.3f}")
            print(f"     {preview.replace(chr(10), ' ')}")
            print()

        print("-" * 60)

if __name__ == "__main__":
    demo_rag_retrieval()