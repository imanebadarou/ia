#!/usr/bin/env python3
"""
Test script for the simplified Dragon Ball RAG system
"""

from simple_rag import SimpleDragonBallRAG

def test_simple_rag():
    """Test the simplified RAG system"""

    # Initialize RAG system
    rag = SimpleDragonBallRAG()

    # Always build index for testing
    print("🔨 Construction de l'index...")
    rag.build_index()
    rag.save_index()
    print("✅ Index construit et sauvegardé!")

    # Test questions - MODIFIEZ CETTE LISTE AVEC VOS QUESTIONS !
    test_questions = [
        "Quelle est la puissance de Frieza ?",
        "Comment Goku devient Super Saiyan ?",
        "Qui est plus fort entre Vegeta et Goku ?",
        "Quelle est l'histoire de Namek ?",
        "Qui sont les membres de la Ginyu Force ?",
        "Quelles sont les Dragon Balls ?",
        "Où se déroule la Frieza Saga ?",
        "Quelle est la planète natale de Vegeta ?"
    ]

    print("\n🧪 Test du système RAG - Dragon Ball")
    print("=" * 50)

    for question in test_questions:
        print(f"\n❓ Question: {question}")
        try:
            result = rag.ask_question(question, k=3)
            print(f"🤖 Réponse: {result['answer']}")
            print(f"📊 Résultats trouvés: {result['num_results']}")
        except Exception as e:
            print(f"❌ Erreur: {e}")
        print("-" * 50)

if __name__ == "__main__":
    test_simple_rag()