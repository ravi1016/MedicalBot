FROM python:3.12-slim

WORKDIR /app

COPY . .

RUN pip install uv
RUN uv sync

EXPOSE 8080

CMD ["uv", "run", "src/app.py"]