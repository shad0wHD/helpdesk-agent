FROM python:3.11-slim

WORKDIR /app

RUN pip install hatch

COPY pyproject.toml .
RUN pip install -e ".[dev]"

COPY . .

# Pre-download the embedding model so the first request isn't slow
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('BAAI/bge-small-en-v1.5')"

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
