import pickle

MODEL_FILE_NAME = "pipeline_v1.bin"
with open(MODEL_FILE_NAME, "rb") as f:
    vectorizer, model = pickle.load(f)

payload = {
    "lead_source": "paid_ads",
    "number_of_courses_viewed": 2,
    "annual_income": 79276.0
}

X = vectorizer.transform([payload])
y_pred = model.predict_proba(X)[0, 1]
print(y_pred)