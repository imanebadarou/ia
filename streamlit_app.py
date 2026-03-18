#!/usr/bin/env python3
"""
Interface Web Streamlit pour le système RAG Dragon Ball
Démonstration interactive du système de questions-réponses
"""

import streamlit as st
import os
from pathlib import Path
from rag_core import DragonBallRAG

# Configuration de la page
st.set_page_config(
    page_title="🤖 Dragon Ball RAG Expert",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Titre principal
st.title("🤖 Dragon Ball RAG Expert")
st.markdown("**Système de Questions-Réponses basé sur votre corpus Dragon Ball**")

# Sidebar pour la configuration
st.sidebar.header("⚙️ Configuration")

# Choix du système RAG
rag_mode = st.sidebar.selectbox(
    "Choisir le système RAG:",
    ["Basique (Rapide)", "Avancé (Reranking)"],
    help="Le système avancé utilise un reranking pour de meilleurs résultats"
)

# Paramètres de recherche
st.sidebar.subheader("🔍 Paramètres de recherche")
k_chunks = st.sidebar.slider("Nombre de chunks à récupérer:", 3, 10, 5)


# Statistiques du corpus
st.sidebar.subheader("📊 Statistiques du Corpus")
try:
    corpus_dir = Path("corpus/saga_freezer")
    if corpus_dir.exists():
        txt_files = list(corpus_dir.glob("*.txt"))
        total_chars = sum(len(f.read_text()) for f in txt_files if f.exists())
        st.sidebar.metric("Documents", len(txt_files))
        st.sidebar.metric("Caractères", f"{total_chars:,}")
        st.sidebar.metric("Chunks estimés", f"{total_chars // 800:,}")
except:
    st.sidebar.error("Corpus non trouvé")

# Fonction pour charger le système RAG
@st.cache_resource
def load_rag_system(mode):
    """Charge le système RAG approprié"""
    try:
        use_reranking = mode == "Avancé (Reranking)"
        vectorstore_path = "advanced_vectorstore" if use_reranking else "vectorstore"

        rag = DragonBallRAG(
            corpus_dir="corpus/saga_freezer",
            vectorstore_path=vectorstore_path,
            use_reranking=use_reranking,
        )

        if not rag.load_index():
            st.error("Index non trouvé. Exécutez le script de construction (test_simple_rag.py) pour générer l'index.")
            return None
        return rag
    except Exception as e:
        st.error(f"Erreur lors du chargement: {e}")
        return None

# Interface principale
def main():
    # Charger le système RAG
    rag = load_rag_system(rag_mode)

    if not rag:
        st.error("Impossible de charger le système RAG. Vérifiez que les index sont construits.")
        return

    # Zone de saisie de question
    st.subheader("❓ Posez votre question sur Dragon Ball")

    question = st.text_input(
        "Votre question:",
        placeholder="Ex: Quelle est la puissance de Frieza ?",
        help="Posez une question sur Dragon Ball et le système trouvera les informations pertinentes dans votre corpus"
    )

    # Bouton de recherche
    if st.button("🔍 Rechercher", type="primary", use_container_width=True):
        if question.strip():
            with st.spinner("Recherche en cours..."):
                try:
                    # Effectuer la recherche
                    if rag_mode == "Avancé (Reranking)":
                        result = rag.ask_question_advanced(question, final_k=k_chunks)
                        search_results = rag.search_with_reranking(question, final_k=k_chunks)
                    else:
                        result = rag.ask_question(question, k=k_chunks)
                        search_results = rag.search(question, k=k_chunks)

                    # Afficher la réponse
                    st.success("Réponse générée !")

                    # Section réponse
                    st.subheader("🤖 Réponse")
                    answer = result['answer']

                    if "Erreur" in answer:
                        st.error(answer)
                    else:
                        st.write(answer)

                    # Section sources
                    st.subheader("📚 Sources utilisées")
                    st.metric("Nombre de chunks analysés", result['num_results'])

                    # Afficher les chunks avec leurs scores
                    for i, chunk_result in enumerate(search_results, 1):
                        with st.expander(f"📄 Source {i} (Score: {chunk_result.get('combined_score', chunk_result.get('score', 'N/A')):.3f})"):
                            # Preview du chunk
                            chunk_text = chunk_result['chunk']
                            if len(chunk_text) > 500:
                                st.write(chunk_text[:500] + "...")
                                if st.button(f"Voir le chunk complet {i}", key=f"full_{i}"):
                                    st.text_area("Chunk complet:", chunk_text, height=200)
                            else:
                                st.write(chunk_text)

                    # Mode évaluation
                    if show_evaluation:
                        st.subheader("🧪 Évaluation")

                        # Charger l'évaluateur
                        evaluator = RAGEvaluator(rag)

                        # Trouver le test case correspondant (approximatif)
                        best_match = None
                        best_score = 0

                        for test_case in evaluator.test_cases:
                            # Similarité simple basée sur les mots
                            q_words = set(question.lower().split())
                            t_words = set(test_case["question"].lower().split())
                            similarity = len(q_words & t_words) / len(q_words | t_words)

                            if similarity > best_score:
                                best_score = similarity
                                best_match = test_case

                        if best_match and best_score > 0.3:
                            st.info(f"Test case trouvé (similarité: {best_score:.2f})")

                            # Évaluation du retrieval
                            retrieval_eval = evaluator.evaluate_retrieval(
                                question,
                                search_results,
                                best_match["key_facts"]
                            )

                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric("Précision Retrieval", f"{retrieval_eval['precision']:.2f}")
                            with col2:
                                st.metric("Rappel Retrieval", f"{retrieval_eval['recall']:.2f}")

                            # Évaluation de la génération
                            if "Erreur" not in answer:
                                generation_eval = evaluator.evaluate_generation(
                                    question,
                                    best_match["expected_answer"],
                                    answer
                                )

                                st.metric("Score Génération", f"{generation_eval['score']}/5")
                                st.write(f"**Justification:** {generation_eval['justification']}")

                                # Comparaison des réponses
                                st.subheader("🔍 Comparaison")
                                col1, col2 = st.columns(2)

                                with col1:
                                    st.write("**Réponse attendue:**")
                                    st.info(best_match["expected_answer"])

                                with col2:
                                    st.write("**Réponse du RAG:**")
                                    st.info(answer)
                        else:
                            st.warning("Aucun test case similaire trouvé pour l'évaluation")

                except Exception as e:
                    st.error(f"Erreur lors de la recherche: {e}")

        else:
            st.warning("Veuillez saisir une question.")

    # Section d'information
    st.markdown("---")
    st.subheader("ℹ️ À propos")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        **Comment ça marche:**
        1. 📚 Votre corpus Dragon Ball est découpé en chunks
        2. 🧮 Chaque chunk est transformé en vecteur (embedding)
        3. 🔍 La question est vectorisée et comparée aux chunks
        4. 🎯 Les chunks les plus pertinents sont sélectionnés
        5. 🤖 Un LLM génère la réponse finale
        """)

    with col2:
        st.markdown("""
        **Technologies utilisées:**
        - **SentenceTransformer** pour les embeddings
        - **FAISS** pour la recherche vectorielle
        - **OpenRouter** pour le LLM
        - **Streamlit** pour l'interface web
        """)

        if rag_mode == "Avancé (Reranking)":
            st.info("🎯 Mode avancé activé: Reranking avec Cross-Encoder pour de meilleurs résultats!")

    # Footer
    st.markdown("---")
    st.markdown("*Construit avec ❤️ pour l'exploration de Dragon Ball*")

if __name__ == "__main__":
    main()