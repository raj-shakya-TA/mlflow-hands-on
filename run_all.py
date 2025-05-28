import mlflow
from scripts import prepare_data, train_model, score_model

with mlflow.start_run(run_name="Housing Pipeline") as parent_run:
    with mlflow.start_run(run_name="Data Preparation", nested=True):
        prepare_data.main()

    with mlflow.start_run(run_name="Model Training", nested=True):
        train_model.main()

    with mlflow.start_run(run_name="Model Scoring", nested=True):
        score_model.main()
