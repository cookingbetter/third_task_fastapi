from fastapi import FastAPI

app = FastAPI(title="ML Churn Service")


@app.get("/")
def read_root():
    return {"message": "ml churn service is running"}
