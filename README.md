# Workflow CI

Push folder ini ke repository public `Workflow-CI`.

Workflow `.github/workflows/train.yml` menjalankan:

1. Install MLflow.
2. `mlflow run MLProject --env-manager=local`.
3. Upload/commit artifact MLflow.
4. Opsional build dan push Docker image jika Docker Hub secrets sudah diisi.

