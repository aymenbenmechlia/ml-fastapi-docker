# ML Forecast API

API de prévision des ventes à 7 jours basée sur **LightGBM**, **BigQuery** et **FastAPI**.

L'application récupère l'historique des ventes depuis BigQuery, construit les variables explicatives nécessaires au modèle de Machine Learning, puis retourne une prévision de quantité pour chacun des 7 prochains jours.

---

## Architecture

```text
                    ┌─────────────────────┐
                    │      BigQuery       │
                    │                     │
                    │ Sales transactions  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │   ml/extract.py     │
                    │                     │
                    │ Data extraction     │
                    │ Feature engineering │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │    LightGBM Models  │
                    │                     │
                    │ 7 daily models      │
                    │ J+1 → J+7           │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │      FastAPI        │
                    │                     │
                    │     /predict       │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │       JSON          │
                    │                     │
                    │ Daily predictions   │
                    │ Total 7 days        │
                    └─────────────────────┘
```

---

## Fonctionnement

Pour un `operational_unit_id` et un `item_id` donnés :

1. L'API interroge BigQuery.
2. Les ventes sont agrégées par jour.
3. Les jours sans vente sont complétés avec une quantité de `0`.
4. Les features temporelles et historiques sont calculées.
5. La dernière ligne disponible est utilisée comme point de prévision.
6. Les 7 modèles LightGBM produisent chacun une prédiction :

   * `target_1d`
   * `target_2d`
   * `target_3d`
   * `target_4d`
   * `target_5d`
   * `target_6d`
   * `target_7d`
7. Les 7 prédictions sont additionnées pour obtenir la quantité totale prévisionnelle sur 7 jours.

---

## Structure du projet

```text
ml-fastapi-docker/
│
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── schemas.py
│   └── config.py
│
├── ml/
│   ├── __init__.py
│   ├── extract.py
│   ├── preprocess.py
│   └── train.py
│
├── models/
│   └── models_7d_optimized.joblib
│
├── data/
│
├── tests/
│   └── __init__.py
│
├── .env
├── .gitignore
├── requirements.txt
└── README.md
```

---

## Prérequis

* Python 3.10+
* Google Cloud SDK
* Accès au projet Google Cloud
* Accès au dataset BigQuery
* Compte Google avec les permissions nécessaires

---

## Installation

Cloner le projet puis se placer dans le répertoire :

```bash
cd ml-fastapi-docker
```

Créer un environnement virtuel :

```bash
python -m venv .venv
```

Activer l'environnement virtuel.

### Windows

```bash
.venv\Scripts\activate
```

### Linux / WSL

```bash
source .venv/bin/activate
```

Installer les dépendances :

```bash
pip install -r requirements.txt
```

---

## Dépendances

Les principales librairies utilisées sont :

```text
pandas
numpy
scikit-learn
lightgbm
joblib
google-cloud-bigquery
db-dtypes
python-dotenv
fastapi
uvicorn
```

---

## Configuration Google Cloud

L'application utilise les **Application Default Credentials (ADC)**.

S'authentifier avec :

```bash
gcloud auth application-default login
```

Le projet Google Cloud utilisé par défaut est :

```text
ddp-dtm-perf-prd-frlm
```

Il peut être configuré dans le fichier `.env` :

```env
GOOGLE_CLOUD_PROJECT=ddp-dtm-perf-prd-frlm
```

---

## Source BigQuery

Les données proviennent de :

```text
ddp-dtm-perf-prd-frlm.base_finance_performance_bu001.int_salesMarginWAC
```

Les principales colonnes utilisées sont :

```text
operationalUnitIdentifier
transactionDate
itemIdentifier
itemQuantity
```

La requête est paramétrée avec :

```text
operational_unit_id
item_id
```

Exemple :

```text
operational_unit_id = 146
item_id = 49016767
```

---

## Feature engineering

Le modèle utilise notamment des variables historiques :

```text
quantity
quantity_lag_1
quantity_lag_7
quantity_lag_14
quantity_lag_28

quantity_mean_7d
quantity_mean_14d
quantity_mean_28d

quantity_std_28d

quantity_sum_7d
quantity_sum_14d
quantity_sum_28d
quantity_sum_56d

quantity_sum_previous_7d
quantity_sum_previous_14d
quantity_sum_previous_28d

ratio_previous_7d_28d
ratio_previous_7d_recent_7d

trend_7d_vs_28d
trend_14d_vs_28d

day_of_week
month
day_of_year

sin_day_of_year
cos_day_of_year

is_weekend
is_holiday
```

Le modèle actuellement sauvegardé utilise également :

```text
baseline_28d_7days
```

Cette variable est reconstruite automatiquement par l'API si elle est présente dans l'artifact du modèle.

---

## Entraînement

L'entraînement est réalisé dans :

```text
ml/train.py
```

Le modèle utilise un split temporel afin d'éviter les fuites de données entre le passé et le futur.

Les données sont séparées en :

```text
Training
    ↓
Gap temporel de 7 jours
    ↓
Test
```

Sept modèles LightGBM sont entraînés pour prévoir chaque horizon :

```text
J+1 → target_1d
J+2 → target_2d
J+3 → target_3d
J+4 → target_4d
J+5 → target_5d
J+6 → target_6d
J+7 → target_7d
```

Les modèles sont sauvegardés dans :

```text
models/models_7d_optimized.joblib
```

---

## Tester l'extraction BigQuery

Depuis la racine du projet :

```bash
python -m ml.extract
```

Un exemple de test utilise :

```python
get_sales_data(
    operational_unit_id=146,
    item_id=49016767,
)
```

Le résultat contient les données historiques ainsi que les features et targets nécessaires à l'entraînement.

---

# API FastAPI

## Démarrer l'API

Depuis la racine du projet :

```bash
uvicorn app.main:app --reload
```

L'API est alors disponible sur :

```text
http://127.0.0.1:8000
```

---

## Swagger

La documentation interactive est disponible à :

```text
http://127.0.0.1:8000/docs
```

Elle permet notamment de tester directement :

```text
GET /health
GET /model-info
POST /predict
```

---

# Endpoints

## GET /health

Vérifie que l'API fonctionne et que le modèle est chargé.

Exemple :

```json
{
  "status": "ok",
  "model_loaded": true
}
```

---

## GET /model-info

Retourne les informations principales sur l'artifact du modèle :

* targets
* target direct
* features
* configuration des modèles
* métriques

Exemple :

```bash
curl http://127.0.0.1:8000/model-info
```

---

## POST /predict

Retourne les prévisions à 7 jours pour un article et une unité opérationnelle.

### Request

```json
{
  "operational_unit_id": 146,
  "item_id": 49016767
}
```

### Response

```json
{
  "operational_unit_id": 146,
  "item_id": 49016767,
  "last_date": "2026-08-25",
  "predictions": {
    "target_1d": 31.41114610907901,
    "target_2d": 37.11063723275201,
    "target_3d": 48.83409818074591,
    "target_4d": 32.63356273117253,
    "target_5d": 16.99193836135299,
    "target_6d": 45.11863949420237,
    "target_7d": 40.35931053422749
  },
  "total_quantity_7d": 252.45933264353232
}
```

La quantité totale prévisionnelle est donc :

```text
252.46 unités
```

---

# Exemple avec Python

```python
import requests

url = "http://127.0.0.1:8000/predict"

payload = {
    "operational_unit_id": 146,
    "item_id": 49016767,
}

response = requests.post(url, json=payload)

print(response.json())
```

---

# Modèle ML

Le modèle principal utilisé est **LightGBM**.

L'approche actuelle utilise plusieurs modèles spécialisés :

```text
                    ┌──────────────┐
                    │ Features     │
                    └──────┬───────┘
                           │
          ┌────────────────┼────────────────┐
          │                │                │
          ▼                ▼                ▼
      Model J+1        Model J+2        Model J+3
          │                │                │
          ▼                ▼                ▼
        Pred 1           Pred 2           Pred 3
          │
          │       ...
          │
          ▼
      Model J+7
          │
          ▼
        Pred 7
          │
          └────────────┬─────────────┘
                       ▼
                 SUM(Pred 1..7)
                       │
                       ▼
                 Total 7 jours
```

Cette approche permet de produire une prédiction spécifique pour chaque horizon temporel.

---

## Modèle sauvegardé

L'artifact :

```text
models/models_7d_optimized.joblib
```

contient notamment :

```text
models
direct_model
targets
direct_target
features
best_configs
direct_best_config
metrics
gap_days
direct_baseline_feature
```

L'API charge cet artifact au démarrage :

```python
model_artifact = joblib.load(MODEL_PATH)
```

Le modèle n'est donc **pas réentraîné à chaque appel API**.

---

# Évaluation

Une baseline basée sur la moyenne des 28 derniers jours a été utilisée comme référence.

Résultat de la baseline :

```text
MAE  : 17.895
RMSE : 24.095
R²   : -0.110
```

Les modèles LightGBM actuels sont encore en phase d'expérimentation et d'optimisation.

L'objectif principal étant la prévision de la **quantité totale sur 7 jours**, l'évaluation doit notamment prendre en compte la performance de l'agrégation des 7 prédictions.

---

# Points importants

### Dernière date disponible

L'API utilise la dernière date disponible dans BigQuery comme point de départ de la prévision.

Par exemple :

```text
Dernière date disponible : 2026-08-25
```

Les prédictions correspondent donc aux jours suivants.

Si les données BigQuery ne sont pas à jour, la date de prévision sera également décalée.

---

### Historique nécessaire

Le modèle utilise des features basées sur plusieurs semaines d'historique.

Un historique suffisant est donc nécessaire pour générer une prédiction fiable.

---

### Valeurs négatives

Les quantités négatives présentes dans les données sont conservées.

Elles peuvent notamment correspondre à des retours ou corrections de ventes.

Elles ne doivent pas être supprimées sans validation métier.

---

# Sécurité

Le fichier `.env` ne doit pas être commité.

Ajouter notamment :

```text
.env
.venv/
__pycache__/
*.pyc
```

dans `.gitignore`.

Les credentials Google Cloud ne doivent jamais être stockés dans le repository.

---

# Développement

Lancer l'API en mode développement :

```bash
uvicorn app.main:app --reload
```

Après modification du code, FastAPI recharge automatiquement l'application grâce à `--reload`.

---

# Tests

Les tests peuvent être placés dans :

```text
tests/
```

Exemple de structure future :

```text
tests/
├── __init__.py
├── test_health.py
├── test_predict.py
└── test_preprocess.py
```

---

# Prochaines améliorations

Les prochaines évolutions prévues sont notamment :

* [ ] Ajouter les dates réelles J+1 à J+7 dans la réponse API
* [ ] Ajouter une réponse Pydantic structurée
* [ ] Gérer les erreurs BigQuery proprement
* [ ] Gérer les données insuffisantes
* [ ] Ajouter des tests unitaires
* [ ] Ajouter Docker
* [ ] Ajouter Docker Compose
* [ ] Ajouter CI/CD
* [ ] Ajouter MLflow pour le suivi des expériences
* [ ] Améliorer les performances du modèle
* [ ] Comparer systématiquement LightGBM avec les baselines
* [ ] Ajouter un monitoring des prédictions
* [ ] Mettre en place une stratégie de retraining

---

# Résumé

Le projet permet actuellement de réaliser le workflow complet :

```text
BigQuery
   ↓
Extraction des ventes
   ↓
Feature engineering
   ↓
LightGBM
   ↓
7 prédictions quotidiennes
   ↓
Agrégation sur 7 jours
   ↓
FastAPI
   ↓
Swagger / API JSON
```

Exemple :

```text
Input
  operational_unit_id = 146
  item_id = 49016767

             ↓

7-day forecast
  J+1 = 31.41
  J+2 = 37.11
  J+3 = 48.83
  J+4 = 32.63
  J+5 = 16.99
  J+6 = 45.12
  J+7 = 40.36

             ↓

Total = 252.46 unités
```

Le projet constitue ainsi une première version fonctionnelle d'une **API de Machine Learning de prévision des ventes**, avec séparation entre extraction des données, preprocessing, entraînement des modèles et exposition via API.
