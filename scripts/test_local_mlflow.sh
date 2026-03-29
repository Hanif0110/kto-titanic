export KUBE_MLFLOW_TRACKING_URI=https://mlflow-$$$$-dev.apps.$$$$$.openshiftapps.com # <--- mettez ici l'url de votre service mlflow
export MLFLOW_TRACKING_URI=https://mlflow-$$$$-dev.apps.$$$$.openshiftapps.com # <--- mettez ici l'url de votre service mlflow
export MLFLOW_S3_ENDPOINT_URL=https://minio-api-$$$$-dev.apps.$$$$.openshiftapps.com # <--- mettez ici l'url de votre service minio
export AWS_ACCESS_KEY_ID=$$$$ # <--- mettez ici le user minio
export AWS_SECRET_ACCESS_KEY=$$$$ # <--- mettez ici le password minio

uv run mlflow run ./src/titanic/training -e main --env-manager=local -P path=all_titanic.csv --experiment-name kto-titanic
