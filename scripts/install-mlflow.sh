#!/usr/bin/env bash

set -euo pipefail

echo "======================================"
echo "Installation de kto-mlflow"
echo "======================================"

echo ""
echo "1) Vérification de la connexion OpenShift..."
oc whoami
oc project

echo ""
echo "2) Vérification des fichiers Kubernetes..."

FILES=(
  "k8s/mlflow/minio.yml"
  "k8s/mlflow/mysql.yml"
  "k8s/mlflow/mlflow.yml"
  "k8s/mlflow/dailyclean.yml"
  "k8s/monitoring/jaeger.yaml"
)

for file in "${FILES[@]}"; do
  if [ ! -f "$file" ]; then
    echo "ERREUR : fichier introuvable -> $file"
    exit 1
  fi
done

echo "Tous les fichiers Kubernetes sont présents."

echo ""
echo "3) Déploiement de MinIO..."
oc apply -f k8s/mlflow/minio.yml

echo ""
echo "4) Déploiement de MySQL..."
oc apply -f k8s/mlflow/mysql.yml

echo ""
echo "5) Déploiement de MLflow..."
oc apply -f k8s/mlflow/mlflow.yml

echo ""
echo "6) Déploiement de DailyClean..."
oc apply -f k8s/mlflow/dailyclean.yml

echo ""
echo "7) Ajout des labels DailyClean..."
oc label deployment dailyclean-api axa.com/dailyclean=false --overwrite || true
oc label statefulset mysql axa.com/dailyclean=true --overwrite || true

echo ""
echo "8) Déploiement de Jaeger..."
oc apply -f k8s/monitoring/jaeger.yaml

echo ""
echo "9) Vérification des ressources créées..."
oc get deployments
oc get statefulsets
oc get pods
oc get routes

echo ""
echo "======================================"
echo "Installation terminée."
echo "Si certains pods ne sont pas encore Running, attends 2-5 minutes puis relance :"
echo "oc get pods"
echo "======================================"