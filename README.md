# Dragon Ball RAG — Interface Streamlit

Une interface **RAG (Retrieval-Augmented Generation)** permettant de poser des questions sur un corpus Dragon Ball local (wiki/saga), et de générer des réponses via OpenRouter.

---

## 🧩 Contenu du projet

- `streamlit_app.py` — Interface web Streamlit
- `rag_core.py` — Logique RAG (index FAISS, recherche + génération via OpenRouter)
- `corpus/saga_freezer/` — Corpus texte (pages Dragon Ball en `.txt`)
- `vectorstore*.faiss`, `*_meta.json` — Index FAISS / métadonnées pour la recherche
- `.env` — Variables d’environnement (clé OpenRouter, etc.)

---

## ⚙️ Prérequis

- Python 3.11+ (tu utilises 3.13)
- `pip` et `venv`
- Connexion internet (pour OpenRouter)
- Clé OpenRouter (recommandé)

---

## ✅ Installation

Dans le dossier `ia/` :

```bash
cd "/Users/imane/Documents/COURS IMAC/ia"

python -m venv .venv
source .venv/bin/activate

pip install -U pip
pip install -r requirements.txt
```

> Si le projet n’a pas de `requirements.txt`, installe les libs suivantes :

```bash
pip install streamlit sentence-transformers faiss-cpu openai python-dotenv
```

---

## 🔑 Configuration (OpenRouter)

Crée un fichier `.env` (ou mets à jour celui existant) :

```env
OPENROUTER_API_KEY=ta_cle_openrouter_personnelle
# Optionnel : changer de modèle pour éviter le modèle "free"
OPENROUTER_MODEL=gpt-4o-mini
```

> ⚠️ Si un message `429` apparaît, c’est généralement un **quota / rate limit OpenRouter** (modèle “free” souvent bloqué).  
> → Utilise ta propre clé + un modèle non gratuit / non partagé.

---

## ▶️ Lancer l’interface

```bash
source .venv/bin/activate
streamlit run streamlit_app.py
```

Ensuite, ouvre l’URL indiquée en sortie (généralement `http://localhost:8501`).

---

## 🧠 Comment ça marche

1. Le script charge le corpus (`corpus/saga_freezer/*.txt`)
2. Il construit un index FAISS (`vectorstore*.faiss`) s’il n’existe pas
3. Il recherche les chunks les plus pertinents à partir de ta question
4. Il envoie ces chunks + ta question à OpenRouter pour générer une réponse

---

## 🔧 Résoudre les erreurs fréquentes

### ❌ `Taux de requêtes dépassé (429)`
- Le modèle “free” est rate-limité.
- Utilise ta propre clé OpenRouter et un modèle stable (`gpt-4o-mini` par exemple).
- Vérifie ton quota dans le dashboard : https://openrouter.ai/

### ❌ `OPENROUTER_API_KEY missing`
- Assure-toi que `.env` contient `OPENROUTER_API_KEY=...`
- Et que tu recharges Streamlit (ferme/re-lance)

### ❌ Index introuvable
- L’index se construit automatiquement si manquant.
- Si ça plante pendant la création : vérifie que `corpus/saga_freezer/*.txt` existe.

---

## 🧪 Ajouter du contenu (Corpus Dragon Ball)

1. Place tes fichiers `.txt` dans `corpus/saga_freezer/`
2. Relance Streamlit → l’index sera reconstruit automatiquement

---

## 📌 Conseils / améliorations possibles

- Ajouter un fallback local (modèle LLaMA via `llama.cpp` ou `gpt4all`)
- Générer un `requirements.txt`
- Ajouter des logs / affichage des scores de recherche
