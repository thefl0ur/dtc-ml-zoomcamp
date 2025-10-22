from pathlib import Path
import requests

MODEL_NAME = "pipeline_v1.bin"
MODELS_SOURCE_URL = f"https://github.com/DataTalksClub/machine-learning-zoomcamp/raw/refs/heads/master/cohorts/2025/05-deployment/{MODEL_NAME}"

def fetch():
    resp = requests.get(
        MODELS_SOURCE_URL,
        timeout=10,
    )
    resp.raise_for_status()
    print(resp.status_code)
    with open(MODEL_NAME, 'wb') as f:
        f.write(resp.content)

if not Path(MODEL_NAME).exists():
    fetch()

