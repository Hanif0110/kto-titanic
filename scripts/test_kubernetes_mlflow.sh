docker login -u="QUAY_ROBOT_USERNAME_A_SAISIR" -p="QUAY_ROBOT_TOKEN_A_SAISIR" quay.io # <--- mettez ici les informations de votre robot quay.io
kubectl config set-cluster openshift-cluster --server=OPENSHIFT_SERVER_A_SAISIR # <--- mettez ici l'url de votre cluster OpenShift
kubectl config set-credentials openshift-credentials --token=secrets.OPENSHIFT_TOKEN_A_SAISIR # <--- mettez ici le token d'accès à votre cluster OpenShift
kubectl config set-context openshift-context --cluster=openshift-cluster --user=openshift-credentials --namespace=vars.OPENSHIFT_USERNAME_A_SAISIR-dev # <--- mettez ici votre namespace OpenShift
kubectl config use openshift-context

export KUBE_MLFLOW_TRACKING_URI=https://mlflow-$$$$-dev.apps.$$$$.openshiftapps.com # <--- mettez ici l'url de votre service mlflow
export MLFLOW_TRACKING_URI=https://mlflow-$$$$-dev.apps.$$$$.openshiftapps.com # <--- mettez ici l'url de votre service mlflow
export MLFLOW_S3_ENDPOINT_URL=https://minio-api-$$$$-dev.apps.$$$$.openshiftapps.com # <--- mettez ici l'url de votre service minio
export AWS_ACCESS_KEY_ID=$$$$
export AWS_SECRET_ACCESS_KEY=$$$$

# Renseignez bien dans la ligne ci-dessous le tag de l'image que vous souhaitez construire (repository quay.io créé au préalable)
docker build -f ./k8s/experiment/Dockerfile -t quay.io/$$$$$/titanic/experiment:latest --build-arg MLFLOW_S3_ENDPOINT_URL=$MINIO_API_ROUTE_URL --build-arg AWS_ACCESS_KEY_ID=$AWS_ACCESS_KEY_ID--build-arg AWS_SECRET_ACCESS_KEY=$AWS_SECRET_ACCESS_KEY ..

uv run mlflow run ../src/titanic/training -P path=all_titanic.csv --experiment-name kto-titanic --backend kubernetes --backend-config ../k8s/experiment/kubernetes_config.json
