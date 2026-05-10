FROM python:3.13-slim

ARG MLFLOW_S3_ENDPOINT_URL
ARG AWS_ACCESS_KEY_ID
ARG AWS_SECRET_ACCESS_KEY

ENV MLFLOW_S3_ENDPOINT_URL=${MLFLOW_S3_ENDPOINT_URL}
ENV AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
ENV AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
ENV GIT_PYTHON_REFRESH=quiet
ENV HOME=/tmp
ENV PYTHONPATH=/app/src

WORKDIR /app

RUN pip install --no-cache-dir uv

COPY pyproject.toml uv.lock README.md .python-version ./
COPY ./src ./src

RUN uv sync -n --all-groups

RUN pip install --no-cache-dir \
    fire \
    pandas==2.3.3 \
    scikit-learn==1.8.0 \
    joblib \
    boto3 \
    ydata-profiling==4.18.0 \
    "mlflow[extras]==3.8.1"