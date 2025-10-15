from fastapi import FastAPI, HTTPException
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd

# FastAPI alkalmazás létrehozása
app = FastAPI()

# Globális változó a modellhez
model = None

# Legjobb modell betöltése MLflow-ból
def load_best_model():
    client = MlflowClient()
    experiment_name = "titanic_exp_2"  # Kísérlet neve
    experiment = client.get_experiment_by_name(experiment_name)
    runs = client.search_runs(experiment.experiment_id)

    best_run = None
    best_auc = 0

    for run in runs:
        auc = run.data.metrics.get("AUC", None)
        if auc is not None and auc > best_auc:
            best_auc = auc
            best_run = run

    if best_run:
        model_uri = f"runs:/{best_run.info.run_id}/model"
        return mlflow.pyfunc.load_model(model_uri)
    else:
        raise Exception("No model with AUC metric found.")

# FastAPI esemény: a modell betöltése az alkalmazás indításakor
@app.on_event("startup")
def startup_event():
    global model
    model = load_best_model()
    print("Model successfully loaded!")

# API útvonalak
@app.get("/")
def read_root():
    return {"message": "Hello! This is a titanic prediction API using PyCaret and MLflow."}

@app.post("/predict")
def predict(data: dict):
    try:
        # Ellenőrizzük, hogy a modell betöltődött-e
        if model is None:
            raise HTTPException(status_code=500, detail="Model not loaded")

        # Adatok betöltése a JSON-ból
        input_data = pd.DataFrame([data])

        # Predikció végrehajtása
        predictions = model.predict(input_data)

        # Válasz formázása
        return {"prediction": predictions.tolist()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
