from pathlib import Path
import pandas as pd
import joblib
from fastapi import FastAPI

from .schemas import PredictionRequest

from google.cloud import bigquery

from .config import PROJECT_ID

from ml.extract import get_sales_data

# Chemin vers le modèle
MODEL_PATH = Path(__file__).resolve().parent.parent / "models" / "models_7d_optimized.joblib"

# Chargement du modèle
model_artifact = joblib.load(MODEL_PATH)

bq_client = bigquery.Client(project=PROJECT_ID)


app = FastAPI(
    title="ML Forecast API",
    description="API de prévision des ventes à 7 jours",
    version="1.0.0",
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": True,
    }
@app.get("/model-info")
def model_info():
    return {
        "targets": model_artifact["targets"],
        "direct_target": model_artifact["direct_target"],
        "features": model_artifact["features"],
        "best_configs": model_artifact["best_configs"],
        "direct_best_config": model_artifact["direct_best_config"],
        "metrics": model_artifact["metrics"],
    }

@app.post("/predict")
def predict(request: PredictionRequest):

    df = get_sales_data(
        operational_unit_id=request.operational_unit_id,
        item_id=request.item_id,
    )

    features = model_artifact["features"]

    latest = df.sort_values("transactionDate").iloc[-1]

    if "baseline_28d_7days" in features:
        latest["baseline_28d_7days"] = latest["quantity_mean_28d"] * 7

    X = latest[features].to_frame().T
    X = X.apply(pd.to_numeric, errors="coerce")

    predictions = {}

    for target, model in model_artifact["models"].items():
        prediction = model.predict(X)[0]
        predictions[target] = float(prediction)

    total_7d = sum(predictions.values())

    return {
        "operational_unit_id": request.operational_unit_id,
        "item_id": request.item_id,
        "last_date": str(latest["transactionDate"]),
        "predictions": predictions,
        "total_quantity_7d": float(total_7d),
    }