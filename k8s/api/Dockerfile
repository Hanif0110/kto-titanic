FROM python:3.13-slim

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock .python-version README.md ./
COPY ./src ./src

RUN uv sync -n --group api

EXPOSE 8080

ENTRYPOINT ["uv", "-n", "run", "--no-sync", "api"]