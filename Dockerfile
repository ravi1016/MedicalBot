FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock* ./
RUN uv sync --no-dev --no-install-project

COPY . .
RUN uv sync --no-dev

EXPOSE 5000

CMD ["uv", "run", "app.py"]