#!/usr/bin/env python3
"""
Évaluation du système RAG Dragon Ball
Mini jeu de test avec métriques de qualité
"""

from simple_rag import SimpleDragonBallRAG
import json
from typing import Dict, List, Any

class RAGEvaluator:
    def __init__(self, rag_system: SimpleDragonBallRAG):
        self.rag = rag_system

        # Jeu de test : questions + réponses attendues
        self.test_cases = [
            {
                "question": "Quelle est la planète natale de Vegeta ?",
                "expected_answer": "Vegeta est originaire de la planète Vegeta, également appelée Planète des Saiyans.",
                "key_facts": ["planète Vegeta", "Planète des Saiyans", "monde natal"]
            },
            {
                "question": "Qui sont les membres de la Ginyu Force ?",
                "expected_answer": "La Ginyu Force est composée de Captain Ginyu, Recoome, Burter, Jeice, et Guldo.",
                "key_facts": ["Captain Ginyu", "Recoome", "Burter", "Jeice", "Guldo"]
            },
            {
                "question": "Qu'est-ce qu'un Super Saiyan ?",
                "expected_answer": "Un Super Saiyan est une transformation légendaire des Saiyans qui multiplie considérablement leur puissance.",
                "key_facts": ["transformation", "Saiyans", "puissance", "légendaire"]
            },
            {
                "question": "Où se déroule la Frieza Saga ?",
                "expected_answer": "La Frieza Saga se déroule principalement sur la planète Namek.",
                "key_facts": ["Namek", "planète Namek"]
            },
            {
                "question": "Quelle est la puissance de Frieza en forme finale ?",
                "expected_answer": "Frieza a une puissance de 530.000 unités dans sa forme finale.",
                "key_facts": ["530.000", "forme finale", "unités"]
            }
        ]

    def evaluate_retrieval(self, question: str, retrieved_chunks: List[Dict[str, Any]], key_facts: List[str]) -> Dict[str, Any]:
        """Évalue la qualité du retrieval"""
        # Vérifier si les chunks contiennent les faits clés
        chunks_text = " ".join([chunk['chunk'] for chunk in retrieved_chunks]).lower()

        found_facts = []
        for fact in key_facts:
            if fact.lower() in chunks_text:
                found_facts.append(fact)

        precision = len(found_facts) / len(retrieved_chunks) if retrieved_chunks else 0
        recall = len(found_facts) / len(key_facts) if key_facts else 0

        return {
            "precision": precision,
            "recall": recall,
            "found_facts": found_facts,
            "total_facts": len(key_facts),
            "chunks_with_info": len(found_facts) > 0
        }

    def evaluate_generation(self, question: str, expected_answer: str, rag_answer: str) -> Dict[str, Any]:
        """Évalue la qualité de la génération (simulation LLM-as-judge)"""
        # Note basée sur des critères simples
        score = 1  # Note de base

        # Vérifier si la réponse contient des éléments de la réponse attendue
        expected_lower = expected_answer.lower()
        rag_lower = rag_answer.lower()

        # Mots clés présents
        key_words = expected_lower.split()
        matching_words = sum(1 for word in key_words if word in rag_lower)
        word_coverage = matching_words / len(key_words) if key_words else 0

        # Erreurs communes à éviter
        has_errors = any(error in rag_lower for error in ["erreur", "désolé", "ne peux pas", "rate limit"])

        if has_errors:
            score = 1
            justification = "Erreur technique détectée"
        elif word_coverage > 0.8:
            score = 5
            justification = "Excellente couverture des informations clés"
        elif word_coverage > 0.6:
            score = 4
            justification = "Bonne couverture, quelques détails manquants"
        elif word_coverage > 0.4:
            score = 3
            justification = "Couverture partielle des informations"
        elif word_coverage > 0.2:
            score = 2
            justification = "Informations limitées"
        else:
            score = 1
            justification = "Informations insuffisantes ou hors sujet"

        return {
            "score": score,
            "justification": justification,
            "word_coverage": word_coverage,
            "has_errors": has_errors
        }

    def run_evaluation(self) -> Dict[str, Any]:
        """Lance l'évaluation complète"""
        print("🧪 Évaluation du système RAG")
        print("=" * 50)

        results = []

        for i, test_case in enumerate(self.test_cases, 1):
            question = test_case["question"]
            expected = test_case["expected_answer"]
            key_facts = test_case["key_facts"]

            print(f"\n📋 Test {i}/{len(self.test_cases)}: {question}")

            # Étape 1: Retrieval
            retrieved_chunks = self.rag.search(question, k=3)
            retrieval_eval = self.evaluate_retrieval(question, retrieved_chunks, key_facts)

            print(f"   📊 Précision: {retrieval_eval['precision']:.3f}")
            print(f"   📊 Rappel: {retrieval_eval['recall']:.3f}")
            print(f"   ✅ Chunks pertinents: {retrieval_eval['chunks_with_info']}")

            # Étape 2: Génération (si pas d'erreur API)
            try:
                rag_result = self.rag.ask_question(question, k=3)
                rag_answer = rag_result['answer']

                if "Erreur" in rag_answer:
                    generation_eval = {
                        "score": 1,
                        "justification": "Erreur API détectée",
                        "word_coverage": 0,
                        "has_errors": True
                    }
                else:
                    generation_eval = self.evaluate_generation(question, expected, rag_answer)

                print(f"   🤖 Score génération: {generation_eval['score']}/5")
                print(f"   💬 Justification: {generation_eval['justification']}")

            except Exception as e:
                generation_eval = {
                    "score": 1,
                    "justification": f"Exception: {e}",
                    "word_coverage": 0,
                    "has_errors": True
                }
                print(f"   ❌ Erreur génération: {e}")

            # Résultat complet
            result = {
                "question": question,
                "expected_answer": expected,
                "retrieval": retrieval_eval,
                "generation": generation_eval
            }
            results.append(result)

        # Statistiques globales
        avg_retrieval_precision = sum(r["retrieval"]["precision"] for r in results) / len(results)
        avg_retrieval_recall = sum(r["retrieval"]["recall"] for r in results) / len(results)
        avg_generation_score = sum(r["generation"]["score"] for r in results) / len(results)

        summary = {
            "total_tests": len(results),
            "avg_retrieval_precision": avg_retrieval_precision,
            "avg_retrieval_recall": avg_retrieval_recall,
            "avg_generation_score": avg_generation_score,
            "results": results
        }

        print(f"\n📊 RÉSULTATS GLOBAUX:")
        print(f"   🔍 Précision retrieval: {avg_retrieval_precision:.3f}")
        print(f"   🔍 Rappel retrieval: {avg_retrieval_recall:.3f}")
        print(f"   🤖 Score génération: {avg_generation_score:.2f}/5")

        return summary

    def save_results(self, results: Dict[str, Any], filename: str = "rag_evaluation.json"):
        """Sauvegarde les résultats d'évaluation"""
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"💾 Résultats sauvegardés dans {filename}")


def main():
    # Initialiser le système RAG
    rag = SimpleDragonBallRAG()

    # Charger l'index existant
    if not rag.load_index():
        print("❌ Index non trouvé. Lancez test_simple_rag.py d'abord.")
        return

    # Initialiser l'évaluateur
    evaluator = RAGEvaluator(rag)

    # Lancer l'évaluation
    results = evaluator.run_evaluation()

    # Sauvegarder les résultats
    evaluator.save_results(results)

    print("\n✅ Évaluation terminée !")


if __name__ == "__main__":
    main()