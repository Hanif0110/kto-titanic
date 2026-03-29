export KUBE_MLFLOW_TRACKING_URI=https://mlflow-hanif0110-dev.apps.rm3.7wse.p1.openshiftapps.com/
export MLFLOW_TRACKING_URI=https://mlflow-hanif0110-dev.apps.rm3.7wse.p1.openshiftapps.com/
export MLFLOW_S3_ENDPOINT_URL=https://minio-api-hanif0110-dev.apps.rm3.7wse.p1.openshiftapps.com
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123

uv run mlflow run ./src/titanic/training -e main --env-manager=local -P path=all_titanic.csv --experiment-name kto-titanic