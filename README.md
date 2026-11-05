# Architecture Transformer : du RNN à GPT et BERT

Ce notebook explique pas à pas comment les modèles de séquences ont évolué, des **RNN** jusqu'aux **Transformers**.

Chaque partie contient une explication simple, un schéma, un code commenté et ses résultats.

---

## Contenu du notebook

| Partie | Modèle | Ce que fait le code |
|---|---|---|
| 1 | **RNN** | Prédit la valeur suivante d'une sinusoïde |
| 2 | **LSTM** | Même tâche, avec une mémoire plus stable |
| 3 | **Transformer** | Même tâche, avec l'auto-attention |
| 4 | **GPT-2** | Génère la suite d'un texte |
| 5 | **BERT** | Devine un mot caché dans une phrase |

---

## 1. Les trois premiers modèles : prédire une sinusoïde

### La tâche

On crée une courbe `sin(x)` de 200 points.  
Le modèle lit **10 valeurs consécutives** et doit prédire **la 11ᵉ**.

- 190 séquences au total
- 150 pour l'entraînement, 40 pour le test (découpage dans l'ordre du temps)
- Perte : `MSELoss` (erreur quadratique moyenne)
- Optimiseur : `Adam`, taux d'apprentissage 0,01
- 300 époques
- Graine aléatoire fixée (`torch.manual_seed(42)`) pour obtenir les mêmes résultats à chaque exécution

### Les modèles

**RNN** : il lit les valeurs une par une et garde une mémoire de 16 nombres.  
Limite : sur de longues séquences, il oublie le début.

**LSTM** : c'est un RNN avec des *portes* (oubli, entrée, sortie).  
Ces portes décident quoi garder, quoi ajouter et quoi oublier.

**Transformer** : il regarde les 10 valeurs **en même temps**.  
Grâce à l'**auto-attention**, chaque position peut tenir compte de toutes les autres.

Architecture utilisée :
- projection de chaque valeur en vecteur de taille 16
- encodage de la position
- 1 bloc encodeur, 2 têtes d'attention, feed-forward de taille 32
- une couche linéaire qui donne la prédiction

### Résultats obtenus

| Modèle | Erreur de test (MSE) | Valeur réelle | Valeur prédite |
|---|---|---|---|
| RNN | 0,000028 | -0,3639 | -0,3601 |
| LSTM | **0,000003** | -0,3639 | -0,3622 |
| Transformer | 0,000006 | -0,3639 | -0,3662 |

**Lecture :** les trois modèles prédisent bien la courbe.  
Ici le LSTM est le plus précis. La sinusoïde est un signal simple et court : l'avantage du Transformer se voit surtout sur de longues séquences et de gros volumes de données.

---

## 2. Modèles préentraînés : GPT-2 et BERT

Ces deux modèles sont des Transformers déjà entraînés sur de grandes quantités de texte. On les télécharge depuis Hugging Face.

### GPT-2 : générer du texte

GPT-2 utilise la partie **décodeur** du Transformer. Il prédit le mot suivant, puis recommence.

- Texte de départ : `"Artificial intelligence can help people"` (6 tokens)
- Paramètres : `max_new_tokens=50`, `temperature=0.8`, `top_p=0.9`
- `temperature` : plus elle est haute, plus le texte est varié
- `top_p` : on ne garde que les mots les plus probables

Exemple de texte généré :

> Artificial intelligence can help people and organizations create jobs and build organizations, which in turn will spur growth and prosperity.

### BERT : compléter une phrase

BERT utilise la partie **encodeur** du Transformer. Il lit la phrase dans les deux sens pour deviner un mot masqué.

- Modèle : `bert-base-multilingual-cased`
- Phrase : `Paris est la [MASK] de la France.` (10 tokens)

Les 5 meilleures propositions :

| Probabilité | Phrase |
|---|---|
| 36,32 % | Paris est la **capitale** de la France. |
| 21,09 % | Paris est la **ville** de la France. |
| 3,16 % | Paris est la **Cité** de la France. |
| 1,67 % | Paris est la **préfecture** de la France. |
| 1,64 % | Paris est la **banlieue** de la France. |

---

## GPT vs BERT en une phrase

| | GPT | BERT |
|---|---|---|
| Partie du Transformer | Décodeur | Encodeur |
| Sens de lecture | De gauche à droite | Dans les deux sens |
| Tâche | Générer la suite | Deviner un mot caché |

---

## Installation (avec uv)

### Prérequis
- Python 3.12
- [uv](https://docs.astral.sh/uv/) installé
- Une connexion internet (GPT-2 et BERT pèsent environ 1,2 Go au premier lancement)

### Étapes

```bash
# 1. Cloner le dépôt
git clone <url-du-depot>
cd <nom-du-dossier>

# 2. Créer l'environnement et installer les dépendances
uv sync
```

Si le projet n'a pas encore de `pyproject.toml` :

```bash
uv init --python 3.12
uv add torch matplotlib transformers ipykernel
```

### Vérifier l'installation

```bash
uv run python -c "import torch, transformers, matplotlib; print('OK')"
```

### Lancer le notebook

**Avec VS Code :** ouvrir `architecture_transformer.ipynb`, puis choisir le noyau `.venv`.

**Avec Jupyter Lab :**

```bash
uv run --with jupyterlab jupyter lab
```

---

## Messages normaux à l'exécution

Ces messages ne sont **pas des erreurs**. Le notebook fonctionne quand même.

**Sous Windows : avertissement sur les *symlinks***  
Hugging Face ne peut pas créer de liens symboliques. Les modèles sont quand même téléchargés, ils prennent juste un peu plus de place.  
Pour masquer le message :

```powershell
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
```

**BERT : tableau `UNEXPECTED`**  
Le modèle téléchargé contient des couches prévues pour une autre tâche (`pooler`, `seq_relationship`). Elles ne servent pas à deviner un mot masqué, donc elles sont ignorées.

---

## Structure du projet

```
.
├── architecture_transformer.ipynb   # Le notebook
├── pyproject.toml                   # Liste des dépendances
├── uv.lock                          # Versions exactes installées
└── README.md
```

---

## Bibliothèques utilisées

- **PyTorch** : créer et entraîner les modèles RNN, LSTM et Transformer
- **Transformers (Hugging Face)** : charger GPT-2 et BERT
- **Matplotlib** : tracer les courbes réelles et prédites

---

