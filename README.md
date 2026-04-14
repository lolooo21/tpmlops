# JupyterProject

Base de projet Python + Jupyter pour exploration de donnees, entrainement de modeles et scripts reutilisables.

## Structure

```text
JupyterProject/
+-- api/                  # API plus tard si necessaire
+-- data/
|   +-- raw/              # Donnees brutes
|   +-- processed/        # Donnees preparees
|   +-- download_data.py  # Charge le CSV dans SQLite
|   +-- load_data.py      # Lit train/test depuis SQLite
+-- model/
|   +-- features.py       # Features temps, distance, trafic, points rares
|   +-- preprocessing.py  # Target et assemblage final des features
|   +-- train.py          # Entrainement simple
|   +-- train_ridge.py    # Modele Ridge proche du notebook taxi
|   +-- test_model.py     # Test rapide d'inference
+-- models/               # Modeles entraines exportes
+-- notebooks/            # Exploration et experimentation
|   +-- 01_exploration.ipynb
|   +-- 02_baseline_and_time_features.ipynb
|   +-- 03_distance_and_map_features.ipynb
+-- reports/
|   +-- figures/          # Graphiques exportes
+-- tests/
+-- common.py             # Chargement de la configuration projet
+-- config.yml            # Parametres et chemins
+-- pyproject.toml        # Dependances et metadata
+-- requirements.txt      # Installation classique
```

## Regles simples

- Garde les notebooks pour l'exploration, la visualisation et les essais.
- Sauvegarde les donnees intermediaires dans `data/processed/`.
- Exporte les modeles dans `models/`.
- Centralise les chemins dans `config.yml`.
- Mets les fonctions de feature engineering dans `model/features.py`.
- Mets la transformation de target et l'assemblage final dans `model/preprocessing.py`.
- Mets les stats descriptives et histogrammes dans `notebooks/`.

## Demarrage

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

## Convention conseillee pour les notebooks

- `01_exploration.ipynb`
- `02_baseline_and_time_features.ipynb`
- `03_distance_and_map_features.ipynb`

## Pipeline actuel

1. `python .\data\download_data.py`
   Charge le CSV taxi dans SQLite et cree les tables `train` et `test`.
2. `python -m model.train`
   Entraine un modele lineaire simple.
3. `python -m model.train_ridge`
   Entraine le modele Ridge avec les features du notebook taxi.
4. `python -m model.test_model`
   Recharge le modele simple et teste quelques predictions.
