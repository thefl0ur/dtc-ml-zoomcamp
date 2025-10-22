import requests

url = "http://0.0.0.0:8000/predict"
client = {
    "lead_source": "organic_search",
    "number_of_courses_viewed": 4,
    "annual_income": 80304.0
}

resp = requests.post(url, json=client)
resp.raise_for_status()
print(resp.json())