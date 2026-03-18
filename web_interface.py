#!/usr/bin/env python3
"""
Interface Web Simplifiée pour le système RAG Dragon Ball
Version sans Streamlit pour éviter les dépendances lourdes
"""

import json
from simple_rag import SimpleDragonBallRAG
from advanced_rag import AdvancedDragonBallRAG
from rag_evaluator import RAGEvaluator

def display_header():
    """Affiche l'en-tête de l'application"""
    print("\n" + "="*80)
    print("🌐 INTERFACE WEB SIMPLIFIÉE - DRAGON BALL RAG EXPERT")
    print("="*80)
    print("🤖 Système de Questions-Réponses basé sur votre corpus Dragon Ball")
    print()

def choose_rag_system():
    """Permet à l'utilisateur de choisir le système RAG"""
    print("🔧 CHOIX DU SYSTÈME RAG:")
    print("1. 🐌 RAG Basique (Rapide)")
    print("2. 🚀 RAG Avancé (Reranking)")
    print("3. 🧪 Mode Évaluation")
    print("4. 📊 Voir les statistiques")
    print("5. ❌ Quitter")
    print()

    while True:
        try:
            choice = input("Votre choix (1-5): ").strip()
            if choice in ['1', '2', '3', '4', '5']:
                return choice
            else:
                print("❌ Choix invalide. Veuillez saisir 1, 2, 3, 4 ou 5.")
        except KeyboardInterrupt:
            return '5'

def load_system(choice):
    """Charge le système approprié"""
    try:
        if choice == '1':
            print("🐌 Chargement du RAG Basique...")
            rag = SimpleDragonBallRAG()
            if not rag.load_index():
                print("❌ Index basique non trouvé. Lancez test_simple_rag.py d'abord.")
                return None, None
            return rag, "basique"

        elif choice == '2':
            print("🚀 Chargement du RAG Avancé...")
            rag = AdvancedDragonBallRAG()
            if not rag.load_index():
                print("❌ Index avancé non trouvé. Lancez advanced_rag.py d'abord.")
                return None, None
            return rag, "avancé"

        elif choice == '3':
            print("🧪 Chargement du système d'évaluation...")
            rag = SimpleDragonBallRAG()
            if not rag.load_index():
                print("❌ Index non trouvé pour l'évaluation.")
                return None, None
            evaluator = RAGEvaluator(rag)
            return evaluator, "evaluation"

        elif choice == '4':
            show_statistics()
            return None, "stats"

        elif choice == '5':
            return None, "quit"

    except Exception as e:
        print(f"❌ Erreur lors du chargement: {e}")
        return None, None

def show_statistics():
    """Affiche les statistiques du corpus et des évaluations"""
    print("\n📊 STATISTIQUES DU SYSTÈME")
    print("-" * 50)

    # Statistiques du corpus
    try:
        from pathlib import Path
        corpus_dir = Path("corpus/saga_freezer")
        if corpus_dir.exists():
            txt_files = list(corpus_dir.glob("*.txt"))
            total_chars = sum(len(f.read_text()) for f in txt_files if f.exists())
            print(f"📚 Documents: {len(txt_files)}")
            print(f"📝 Caractères totaux: {total_chars:,}")
            print(f"✂️ Chunks estimés: {total_chars // 800:,}")
        else:
            print("❌ Corpus non trouvé")
    except Exception as e:
        print(f"❌ Erreur corpus: {e}")

    # Statistiques d'évaluation
    try:
        if Path("rag_evaluation.json").exists():
            with open("rag_evaluation.json", 'r', encoding='utf-8') as f:
                eval_data = json.load(f)

            print(f"\n🧪 Évaluation ({eval_data['total_tests']} tests):")
            print(".3f"            print(".3f"            print(".1f"        else:
            print("\n🧪 Aucune évaluation trouvée")
    except Exception as e:
        print(f"❌ Erreur évaluation: {e}")

    print("\nAppuyez sur Entrée pour continuer...")
    input()

def ask_question_interface(rag, system_type):
    """Interface pour poser des questions"""
    print(f"\n❓ MODE QUESTION - {system_type.upper()}")
    print("Tapez votre question ou 'retour' pour revenir au menu")
    print("-" * 60)

    while True:
        question = input("\n🤔 Votre question: ").strip()

        if question.lower() in ['retour', 'back', 'menu']:
            break

        if question:
            try:
                with console_spinner("🔍 Recherche en cours..."):
                    if system_type == "avancé":
                        result = rag.ask_question_advanced(question)
                        search_results = rag.search_with_reranking(question, final_k=5)
                    else:
                        result = rag.ask_question(question, k=5)
                        search_results = rag.search(question, k=5)

                # Afficher la réponse
                print("\n🤖 RÉPONSE:"                print("-" * 40)
                answer = result['answer']
                if "Erreur" in answer:
                    print(f"❌ {answer}")
                else:
                    print(answer)

                # Afficher les sources
                print(f"\n📚 SOURCES ({len(search_results)} chunks):")
                print("-" * 40)
                for i, chunk_result in enumerate(search_results, 1):
                    score = chunk_result.get('combined_score', chunk_result.get('score', 'N/A'))
                    print(f"{i}. Score: {score:.3f}")
                    preview = chunk_result['chunk'][:150].replace('\n', ' ')
                    print(f"   {preview}...")
                    print()

            except Exception as e:
                print(f"❌ Erreur: {e}")
        else:
            print("❌ Veuillez saisir une question.")

def evaluation_interface(evaluator):
    """Interface pour l'évaluation"""
    print("\n🧪 MODE ÉVALUATION")
    print("Évaluez la qualité du système RAG")
    print("-" * 50)

    choice = input("Lancer l'évaluation complète ? (o/n): ").strip().lower()
    if choice in ['o', 'oui', 'yes', 'y']:
        print("🔬 Lancement de l'évaluation...")
        results = evaluator.run_evaluation()
        evaluator.save_results(results)

        print("\n📊 RÉSULTATS DÉTAILLÉS:")
        print("-" * 30)
        for i, result in enumerate(results['results'], 1):
            print(f"Test {i}: {result['question']}")
            print(".3f"            print(".3f"            print(f"  Génération: {result['generation']['score']}/5")
            print()

    print("Appuyez sur Entrée pour continuer...")
    input()

def console_spinner(message):
    """Simple spinner pour la console"""
    import time
    import threading

    def spin():
        spinner = ['⠋', '⠙', '⠹', '⠸', '⠼', '⠴', '⠦', '⠧', '⠇', '⠏']
        i = 0
        while not done:
            print(f'\r{message} {spinner[i % len(spinner)]}', end='', flush=True)
            time.sleep(0.1)
            i += 1
        print('\r' + ' ' * (len(message) + 2) + '\r', end='', flush=True)

    done = False
    spinner_thread = threading.Thread(target=spin)
    spinner_thread.start()

    try:
        yield
    finally:
        done = True
        spinner_thread.join()

def main():
    """Fonction principale"""
    display_header()

    while True:
        choice = choose_rag_system()
        system, system_type = load_system(choice)

        if system_type == "quit":
            print("\n👋 Au revoir !")
            break
        elif system_type == "stats":
            continue
        elif system_type == "evaluation":
            evaluation_interface(system)
        elif system and system_type in ["basique", "avancé"]:
            ask_question_interface(system, system_type)
        else:
            print("❌ Système non disponible. Réessayez.")

if __name__ == "__main__":
    main()