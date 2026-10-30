FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md ./

RUN uv sync --frozen --no-dev --group inference --no-install-project

COPY src ./src

RUN uv sync --frozen --no-dev --group inference

ENV PYTHONPATH=/app/src

EXPOSE 8000

CMD ["uv", "run", "--no-dev", "--group", "inference", "uvicorn", "autopassai.main:app", "--host", "0.0.0.0", "--port", "8000"]
