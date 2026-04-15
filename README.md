# JupyterProject

Base de projet Python + Jupyter pour exploration de donnees, entrainement de modeles, inference locale et API FastAPI.

## Structure

```text
JupyterProject/
+-- api/                  # API FastAPI pour l'inference
|   +-- main.py          # Endpoints HTTP
|   +-- schemas.py       # Schemas Pydantic request/response
|   +-- validation.py    # Validation geographique des requetes
|   +-- model_metadata.py # Metadonnees du modele charge par l'API
|   +-- model_registry.py # Resolution et chargement des versions de modele
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
- `models/taxi_trip_duration_custom.metadata.json`
- `models/custom_versions/<model_version>.model`
- `models/custom_versions/<model_version>.metadata.json`

## Test rapide du modele custom

```powershell
.\.venv\Scripts\python.exe -m model.evaluation.test_model
```

Ce script recharge `models/taxi_trip_duration_custom.model` et affiche quelques predictions sur des lignes aleatoires du jeu de test.

## Fonctionnement de l'API

L'API repose sur 6 couches simples:

1. `api/main.py`
   Expose les endpoints FastAPI.
2. `api/schemas.py`
   Valide les payloads d'entree et les reponses avec Pydantic.
3. `api/validation.py`
   Regroupe les regles geographiques appliquees avant l'inference.
4. `api/model_registry.py`
   Selectionne la version demandee et charge le bon artefact de modele.
5. `api/service.py`
   Charge le modele, appelle la prediction et orchestre la sauvegarde.
6. `api/repository.py`
   Cree les tables SQLite si besoin et persiste predictions + metadata du modele.

Flux d'une requete `POST /predict`:

1. FastAPI recoit le JSON et le valide avec `TripPredictionRequest`.
2. `PredictionService` convertit le payload en `DataFrame`.
3. Le modele custom `TaxiTripDurationModel` applique preprocessing + prediction + postprocessing.
4. `PredictionRepository` enregistre la prediction avec l'horodatage d'inference et la version du modele.
5. L'API retourne `prediction_id` et `trip_duration`.

## Versioning des modeles

Chaque entrainement du modele custom genere:
- une nouvelle version horodatee
- un artefact versionne dans `models/custom_versions/`
- une metadata JSON versionnee
- un alias `latest` via `models/taxi_trip_duration_custom.model`
- un alias `latest` via `models/taxi_trip_duration_custom.metadata.json`

Dans l'API:
- `model_version` est optionnel sur `/predict`
- `model_version` est optionnel sur `/predict_batch`
- si `model_version` est absent, l'API utilise la derniere version disponible
- si `model_version` est fourni, l'API charge explicitement cette version

## Logging de metadonnees

L'API conserve aussi des informations de traçabilite dans SQLite.

Table `model_metadata`:
- `version`
- `model_path`
- `model_created_at`
- `registered_at`

Table `predictions`:
- `inference_timestamp`
- `model_version`
- payload d'entree
- prediction

La version du modele n'est pas stockee dans `config.yml`.
Chaque entrainement genere automatiquement le fichier `models/taxi_trip_duration_custom.metadata.json`.
L'API lit ce fichier pour connaitre la version exacte du modele charge.

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
- `POST /predict_batch`
- `GET /docs`

## Interface Streamlit

Une interface graphique Streamlit est fournie dans `ui/app.py`.

Fonctionnalites:
- choix du `pickup` et du `dropoff` par clic sur une carte
- ou recherche d'adresses new-yorkaises avec suggestions
- saisie de la date via `st.date_input` et de l'heure via `st.time_input`
- saisie du nombre de passagers via `st.slider`
- affichage du payload envoye a l'API
- appel direct a `POST /predict`
- affichage de la duree predite et de la version de modele utilisee
- gestion de messages d'erreur utilisateur pour les inputs invalides et les versions de modele inconnues

Fichiers lies a l'interface:
- `ui/app.py` : application Streamlit principale
- `streamlit_app.py` : wrapper de compatibilite qui lance `ui.app.main()`
- `api/main.py` : endpoint HTTP `POST /predict` appele par l'UI
- `api/schemas.py` : format exact du payload et de la reponse
- `api/error_handlers.py` : structure des erreurs transformees ensuite en messages lisibles dans l'UI

Flux des donnees dans l'interface:
1. L'utilisateur saisit les attributs dans les widgets Streamlit:
   `st.date_input`, `st.time_input`, `st.slider`, `st.number_input`, `st.selectbox`.
2. Les coordonnees sont renseignees soit:
   par clic sur la carte, stocke en `st.session_state`,
   soit par recherche d'adresse, puis conversion de la suggestion choisie en latitude/longitude.
3. `build_payload()` assemble toutes les valeurs UI en un dictionnaire Python conforme a `TripPredictionRequest`.
4. `call_prediction_api()` envoie ce dictionnaire en JSON via `requests.post()` vers `POST /predict`.
5. L'API retourne `prediction_id`, `trip_duration` et `model_version`.
6. L'UI formate `trip_duration` en `hh:mm:ss` puis affiche le resultat.
7. Si l'API renvoie une erreur, `extract_api_error_message()` convertit la reponse JSON en message utilisateur plus lisible.

Extrait simplifie du payload envoye par l'UI:

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

Installation des dependances UI:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Lancement conseille dans 2 terminaux:

Terminal 1, API:

```powershell
.\.venv\Scripts\python.exe -m uvicorn api.main:app --reload
```

Terminal 2, Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\ui\app.py
```

Si tu vois `ModuleNotFoundError: No module named 'streamlit_folium'`, cela signifie que les dependances UI ne sont pas encore installees dans le meme virtualenv que celui utilise pour lancer Streamlit.

Si tu vois `No module named pip`, recrée d'abord `pip` dans le venv:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Remarque:
- l'autocompletion d'adresse s'appuie sur un service de geocodage OpenStreetMap/Nominatim
- pour un vrai autocomplete "temps reel" en production, il vaut mieux passer par Mapbox, Google Places ou Algolia Places selon les contraintes produit et quota

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

Exemple avec version explicite:

- `POST /predict?model_version=taxi_trip_duration_custom_20260415T120000Z`

Exemple de body pour `POST /predict_batch`:

```json
{
  "trips": [
    {
      "vendor_id": 1,
      "pickup_datetime": "2016-03-14T17:24:55",
      "passenger_count": 1,
      "pickup_longitude": -73.982154,
      "pickup_latitude": 40.767937,
      "dropoff_longitude": -73.96463,
      "dropoff_latitude": 40.765602,
      "store_and_fwd_flag": "N"
    },
    {
      "vendor_id": 2,
      "pickup_datetime": "2016-03-14T18:02:10",
      "passenger_count": 2,
      "pickup_longitude": -73.99012,
      "pickup_latitude": 40.75021,
      "dropoff_longitude": -73.97845,
      "dropoff_latitude": 40.76111,
      "store_and_fwd_flag": "N"
    }
  ]
}
```

Exemple de reponse:

```json
{
  "prediction_id": 3,
  "trip_duration": 480,
  "model_version": "taxi_trip_duration_custom_20260415T120000Z"
}
```

Exemple de reponse pour `POST /predict_batch`:

```json
{
  "model_version": "taxi_trip_duration_custom_20260415T120000Z",
  "predictions": [
    {
      "prediction_id": 4,
      "trip_duration": 480
    },
    {
      "prediction_id": 5,
      "trip_duration": 615
    }
  ]
}
```

Chaque appel a `/predict` ajoute une ligne dans la table SQLite `predictions`.
Le modele charge est aussi reference dans la table SQLite `model_metadata`.

## TODO Logging

Ameliorations possibles pour le logging d'inference:
- stocker le payload brut complet de la requete en JSON
- stocker les inputs valides apres validation
- stocker les features calculees utilisees par le modele pour chaque prediction
- stocker la liste des features attendues par version de modele
- stocker les metriques du modele par version: train, validation, test
- stocker le temps d'inference pour chaque requete
- stocker les erreurs de prediction et de validation dans un journal dedie
- ajouter un endpoint simple pour consulter les metadata du modele et l'historique des predictions

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

### `ModuleNotFoundError: No module named 'streamlit_folium'`

L'application Streamlit depend de `streamlit-folium` pour gerer la carte interactive.

Installe les dependances dans le venv du projet:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Puis relance Streamlit:

```powershell
.\.venv\Scripts\python.exe -m streamlit run .\ui\app.py
```

### `No module named pip`

Si cette erreur apparait dans `.venv`, cela signifie que `pip` n'est pas present dans cet environnement Python. Reinstalle-le avec:

```powershell
.\.venv\Scripts\python.exe -m ensurepip --upgrade
```

Ensuite installe ou mets a jour les dependances:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

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
