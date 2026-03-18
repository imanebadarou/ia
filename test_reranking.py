#!/usr/bin/env python3
"""
Test rapide du reranking avancé
"""

from advanced_rag import AdvancedDragonBallRAG

def test_reranking():
    """Test rapide du système de reranking"""
    print("🎯 Test du Reranking Avancé")
    print("=" * 40)

    # Initialiser le système
    rag = AdvancedDragonBallRAG()

    # Toujours construire l'index pour le test
    print("🔨 Construction de l'index avancé...")
    rag.build_index()
    rag.save_index()
    print("✅ Index avancé construit et sauvegardé!")

    # Question de test
    question = "Qui sont les membres de la Ginyu Force ?"

    print(f"❓ Question: {question}")

    # Recherche avec reranking
    print("\n🔍 Recherche avec reranking...")
    results = rag.search_with_reranking(question, initial_k=10, final_k=3)

    print(f"\n📋 Top {len(results)} résultats rerankés:")
    for i, result in enumerate(results, 1):
        print(f"{i}. Score combiné: {result['combined_score']:.4f}")
        print(f"   Score initial: {result['initial_score']:.4f}")
        print(f"   Score rerank: {result['rerank_score']:.4f}")
        preview = result['chunk'][:120].replace('\n', ' ')
        print(f"   Preview: {preview}...")
        print()

    print("✅ Test reranking terminé!")

if __name__ == "__main__":
    test_reranking()