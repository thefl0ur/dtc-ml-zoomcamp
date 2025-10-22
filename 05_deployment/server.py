import pickle
import os

from fastapi import FastAPI
from pydantic import BaseModel

class Customer(BaseModel):
    lead_source: str
    number_of_courses_viewed: int
    annual_income: float

def _load_model():
    MODEL_FILE_NAME = f"pipeline_v{os.getenv('VERSION', '1')}.bin"

    with open(MODEL_FILE_NAME, "rb") as f:
        vectorizer, model = pickle.load(f)

    return vectorizer, model

app = FastAPI()
vectorizer, model = _load_model()


@app.post("/predict")
def read_root(customer: Customer):
    X = vectorizer.transform([customer.model_dump()])
    y_pred = model.predict_proba(X)[0, 1]
    return {"prediction": y_pred}