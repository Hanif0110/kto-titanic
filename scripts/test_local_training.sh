export MLFLOW_S3_ENDPOINT_URL=https://minio-api-hanif0110-dev.apps.rm3.7wse.p1.openshiftapps.com
export AWS_ACCESS_KEY_ID=minio
export AWS_SECRET_ACCESS_KEY=minio123
uv run ./src/titanic/training/main.py --input_data_path "all_titanic.csv" --n_estimators 100 --max_depth 10 --random_state 42