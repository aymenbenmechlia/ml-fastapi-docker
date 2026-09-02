# ml-fastapi-docker

API de prédiction ML servie avec FastAPI et packagée avec Docker.

## Arborescence

```
ml-fastapi-docker/
├── app/          # API FastAPI (main, schemas, model, config)
├── ml/           # Pipeline ML (extract, preprocess, train, evaluate)
├── models/       # Artefacts entraînés (model.joblib)
├── data/         # Jeux de données (sample.csv)
├── tests/        # Tests de l'API
├── requirements.txt
├── Dockerfile
├── docker-compose.yml
└── .env
```

## Démarrage local

```bash
pip install -r requirements.txt
python -m ml.train          # génère models/model.joblib
uvicorn app.main:app --reload
```

## Avec Docker

```bash
docker compose up --build
```

## Endpoints

- `GET /health` — état du service
- `POST /predict` — `{"features": [1.0, 2.0, 3.0]}`

## Tests

```bash
pytest
```
