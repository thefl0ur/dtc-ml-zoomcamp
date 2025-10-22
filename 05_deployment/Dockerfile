FROM agrigorev/zoomcamp-model:2025 AS base

COPY pyproject.toml server.py ./

RUN pip install -e .

ENV VERSION=2

CMD ["uvicorn", "server:app", "--host", "0.0.0.0"]