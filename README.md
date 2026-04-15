# JupyterProject

Base de projet Python + Jupyter pour exploration de donnees, entrainement de modeles, inference locale et API FastAPI.

## Structure

```text
JupyterProject/
+-- api/                  # API FastAPI pour l'inference
|   +-- main.py          # Endpoints HTTP
|   +-- schemas.py       # Schemas Pydantic request/response
|   +-- validation.py    # Validation geographique des requetes
|   +-- service.py       # Orchestration prediction + persistence
|   +-- repository.py    # Acces SQLite pour les predictions
|   +-- config.py        # Parametres de l'API
+-- data/
|   +-- raw/              # Donnees brutes
|   +-- processed/        # Donnees preparees
|   +-- download_data.py  # Charge le CSV dans SQLite
|   +-- load_data.py      # Lit train/test depuis SQLite
+-- model/
|   +-- training/
|   |   +-- train.py                # Entrainement simple
|   |   +-- train_ridge.py          # Pipeline Ridge
|   |   +-- train_custom_model.py   # Wrapper custom pour l'API
|   +-- inference/
|   |   +-- custom_model.py         # Artefact d'inference serialize
|   +-- preprocessing/
|   |   +-- features.py             # Features temps, distance, trafic, points rares
|   |   +-- preprocessing.py        # Target et assemblage final des features
|   |   +-- ridge_features.py       # Colonnes partagees train/inference
|   +-- evaluation/
|   |   +-- test_model.py           # Test rapide d'inference
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
- Mets les fonctions de feature engineering dans `model/preprocessing/features.py`.
- Mets la transformation de target et l'assemblage final dans `model/preprocessing/preprocessing.py`.
- Mets les stats descriptives et histogrammes dans `notebooks/`.
- Garde `api/main.py` le plus fin possible: HTTP dans `main`, logique metier dans `service`, SQLite dans `repository`.

## Demarrage

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
jupyter lab
```

## Preparation des donnees

```powershell
.\.venv\Scripts\python.exe .\data\download_data.py
```

Ce script:
- lit le CSV taxi depuis `data/raw/`
- cree les tables `train` et `test` dans `data/taxi_trip_duration.db`

## Entrainement des modeles

Modele lineaire simple:

```powershell
.\.venv\Scripts\python.exe -m model.training.train
```

Modele Ridge avec pipeline sklearn:

```powershell
.\.venv\Scripts\python.exe -m model.training.train_ridge
```

Modele custom consomme par l'API:

```powershell
.\.venv\Scripts\python.exe -m model.training.train_custom_model
```

Artefacts produits:
- `models/taxi_trip_duration.model`
- `models/taxi_trip_duration_ridge.model`
- `models/taxi_trip_duration_custom.model`

## Test rapide du modele custom

```powershell
.\.venv\Scripts\python.exe -m model.evaluation.test_model
```

Ce script recharge `models/taxi_trip_duration_custom.model` et affiche quelques predictions sur des lignes aleatoires du jeu de test.

## Fonctionnement de l'API

L'API repose sur 4 couches simples:

1. `api/main.py`
   Expose les endpoints FastAPI.
2. `api/schemas.py`
   Valide les payloads d'entree et les reponses avec Pydantic.
3. `api/validation.py`
   Regroupe les regles geographiques appliquees avant l'inference.
4. `api/service.py`
   Charge le modele, appelle la prediction et orchestre la sauvegarde.
5. `api/repository.py`
   Cree la table `predictions` si besoin et persiste chaque prediction dans SQLite.

Flux d'une requete `POST /predict`:

1. FastAPI recoit le JSON et le valide avec `TripPredictionRequest`.
2. `PredictionService` convertit le payload en `DataFrame`.
3. Le modele custom `TaxiTripDurationModel` applique preprocessing + prediction + postprocessing.
4. `PredictionRepository` enregistre l'entree et la prediction dans la table `predictions`.
5. L'API retourne `prediction_id` et `trip_duration`.

## Validation des inputs

Avant d'appeler le modele, l'API valide les coordonnees du trajet.

Regles appliquees sur `POST /predict`:
- `pickup_longitude` et `dropoff_longitude` doivent rester dans la bounding box NYC configuree.
- `pickup_latitude` et `dropoff_latitude` doivent rester dans la bounding box NYC configuree.
- la distance Haversine entre pickup et dropoff doit etre strictement superieure au seuil configure

Parametres de validation:
- `config.yml > api.validation.min_trip_distance_meters`
- `config.yml > api.validation.nyc_bounding_box.longitude`
- `config.yml > api.validation.nyc_bounding_box.latitude`

La formule de Haversine calcule une distance "a vol d'oiseau" entre pickup et dropoff. Ici, elle sert juste a filtrer les courses quasi nulles avant prediction.

## Lancer l'API

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Endpoints utiles:
- `GET /`
- `GET /health`
- `GET /test/random`
- `POST /predict`
- `GET /docs`

## Exemple de prediction

Ouvrir Swagger:

- `http://127.0.0.1:8000/docs`

Exemple de body pour `POST /predict`:

```json
{
  "vendor_id": 1,
  "pickup_datetime": "2016-03-14T17:24:55",
  "passenger_count": 1,
  "pickup_longitude": -73.982154,
  "pickup_latitude": 40.767937,
  "dropoff_longitude": -73.96463,
  "dropoff_latitude": 40.765602,
  "store_and_fwd_flag": "N"
}
```

Exemple de reponse:

```json
{
  "prediction_id": 3,
  "trip_duration": 480
}
```

Chaque appel a `/predict` ajoute une ligne dans la table SQLite `predictions`.

## Troubleshooting

### `GET /` retourne `404 Not Found`

Si l'API tourne mais que `http://127.0.0.1:8000/` retourne `404`, verifie que tu lances bien la version actuelle:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Tu peux aussi tester directement:
- `http://127.0.0.1:8000/docs`
- `http://127.0.0.1:8000/health`

### Le modele custom n'existe pas encore

Si l'API ne trouve pas `models/taxi_trip_duration_custom.model`, il faut regenerer l'artefact:

```powershell
.\.venv\Scripts\python.exe -m model.training.train_custom_model
```

### Erreur de type `TaxiTripDurationModel` ou probleme de `pickle`

Si tu vois une erreur du style:
- `Expected a TaxiTripDurationModel artifact`
- `ModuleNotFoundError` au chargement du modele

cela signifie souvent que l'artefact a ete cree avec une ancienne structure de fichiers. Regenerer simplement le modele custom:

```powershell
.\.venv\Scripts\python.exe -m model.training.train_custom_model
```

### La table `predictions` n'existe pas

La table est creee automatiquement par `PredictionRepository` au chargement du service. Si tu obtiens une erreur SQLite sur `predictions`, relance proprement l'API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

### La prediction echoue car les donnees SQLite n'existent pas

Si les tables `train` ou `test` n'existent pas encore dans `data/taxi_trip_duration.db`, il faut recharger les donnees:

```powershell
.\.venv\Scripts\python.exe .\data\download_data.py
```

### `favicon.ico` retourne `404`

C'est normal tant qu'aucune icone n'est servie par l'application. Cela n'indique pas un probleme de fonctionnement de l'API.

## Convention conseillee pour les notebooks

- `01_exploration.ipynb`
- `02_baseline_and_time_features.ipynb`
- `03_distance_and_map_features.ipynb`

## Workflow conseille

1. `.\.venv\Scripts\python.exe .\data\download_data.py`
   Prepare la base SQLite avec `train` et `test`.
2. `.\.venv\Scripts\python.exe -m model.training.train_custom_model`
   Genere l'artefact custom utilise par l'API.
3. `.\.venv\Scripts\python.exe -m model.evaluation.test_model`
   Verifie rapidement le comportement du modele.
4. `.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload`
   Lance l'API locale.
