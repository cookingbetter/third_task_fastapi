from fastapi import FastAPI

from schemas import FeatureVectorChurn

app = FastAPI(title="ML Churn Service")


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
def predict(features: FeatureVectorChurn):
    return features
