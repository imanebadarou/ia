# Dragon Ball RAG System

Un système de Retrieval-Augmented Generation (RAG) pour répondre aux questions sur Dragon Ball en utilisant votre corpus de pages wiki.

## Architecture

Le système suit les étapes classiques d'un RAG :

### I) Ingestion — Préparation des données
- ✅ Chargement du corpus (pages wiki Dragon Ball)
- ✅ Découpage en chunks (~1000 tokens avec chevauchement)
- ✅ Vectorisation avec SentenceTransformer (all-MiniLM-L6-v2)
- ✅ Stockage dans FAISS (base vectorielle)

### II) Retrieval — Recherche d'information
- ✅ Vectorisation de la question utilisateur
- ✅ Recherche des chunks les plus similaires (cosine similarity)
- ✅ Récupération des K meilleurs résultats

### III) Génération — Réponse intelligente
- ✅ Construction d'un prompt avec contexte
- ✅ Appel à OpenRouter API (modèles Llama)
- ✅ Réponse précise basée uniquement sur le corpus

## Fichiers

- `simple_rag.py` - Système RAG principal
- `test_simple_rag.py` - Test complet avec génération
- `demo_rag.py` - Démo de la recherche seule
- `rag_system.py` - Version avancée avec LangChain (lourde)
- `wiki_downloader.py` - Outil de téléchargement des pages wiki

## Installation

```bash
# Activer l'environnement virtuel
cd "/Users/imane/Documents/COURS IMAC/ia"
source .venv/bin/activate

# Installer les dépendances
pip install sentence-transformers faiss-cpu openai python-dotenv
```

## Configuration

1. **Clé API OpenRouter** : Ajouter dans `.env`
   ```
   OPENROUTER_API_KEY=votre_clé_ici
   ```

2. **Corpus** : Pages wiki dans `corpus/saga_freezer/`

## Utilisation

### Construction de l'index
```bash
python test_simple_rag.py  # Construit l'index et teste
```

### Démo de recherche
```bash
python demo_rag.py  # Montre la recherche sans API
```

### Interface interactive
```bash
python simple_rag.py  # Chatbot RAG complet
```

## Résultats

Le système a été testé avec succès :
- **14 documents** chargés (667,000+ caractères)
- **1007 chunks** créés
- **Embeddings** générés (dimension 384)
- **Recherche précise** avec scores de similarité

## Exemples de questions

- "Qui est Frieza ?"
- "Qu'est-ce qu'un Super Saiyan ?"
- "Où se déroule la Frieza Saga ?"
- "Qui sont les membres de la Ginyu Force ?"

## Métriques

- **Précision de recherche** : Excellente (scores > 0.4)
- **Rappels pertinents** : Tous les chunks trouvés sont contextuellement appropriés
- **Performance** : ~10 secondes pour construire l'index

## Limites actuelles

- API OpenRouter limitée (rate limits sur modèles gratuits)
- Nécessite une clé API pour la génération
- Corpus limité à la Frieza Saga

## Améliorations possibles

- Ajouter plus de documents au corpus
- Utiliser des modèles locaux (Llama.cpp, Ollama)
- Implémenter la recherche hybride (BM25 + vectorielle)
- Ajouter une interface web (Streamlit, Gradio)