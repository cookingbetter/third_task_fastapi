from typing import Annotated

import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query

from dataset import get_dataset
from schemas import DatasetInfo, DatasetRowChurn, FeatureVectorChurn

app = FastAPI(title="ML Churn Service")


def dataset_dependency() -> pd.DataFrame:
    try:
        return get_dataset()
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="dataset file not found")


Dataset = Annotated[pd.DataFrame, Depends(dataset_dependency)]


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}


@app.post("/predict", response_model=FeatureVectorChurn)
def predict(features: FeatureVectorChurn):
    return features


@app.get("/dataset/preview", response_model=list[DatasetRowChurn])
def dataset_preview(df: Dataset, n: int = Query(5, ge=1, le=100)):
    return df.head(n).to_dict(orient="records")


@app.get("/dataset/info", response_model=DatasetInfo)
def dataset_info(df: Dataset):
    counts = df["churn"].value_counts().sort_index()
    return DatasetInfo(
        n_rows=len(df),
        n_columns=len(df.columns),
        features=list(df.columns),
        churn_distribution={int(k): int(v) for k, v in counts.items()},
        churn_share={int(k): round(v / len(df), 4) for k, v in counts.items()},
    )
