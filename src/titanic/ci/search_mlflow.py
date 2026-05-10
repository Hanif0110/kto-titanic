import logging

import fire
import mlflow
from mlflow.entities import Run


def get_last_model_uri(experiment_name: str) -> str:
    logging.warning("experiment_name: %s", experiment_name)

    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        raise ValueError(f"Experiment not found: {experiment_name}")

    runs: list[Run] = mlflow.search_runs(
        [experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'",
        max_results=1,
        order_by=["attributes.end_time DESC"],
        output_format="list",
    )

    if not runs:
        raise ValueError(f"No finished run found for experiment: {experiment_name}")

    run = mlflow.get_run(runs[0].info.run_id)

    if getattr(run.outputs, "model_outputs", None):
        model_id = run.outputs.model_outputs[0].model_id
        model_uri = f"models:/{model_id}"
        logging.warning("Found registered model id: %s", model_id)
    else:
        model_uri = f"runs:/{run.info.run_id}/model_final"
        logging.warning("No model id found. Fallback to run artifact: %s", model_uri)

    logging.warning("Returning: %s", model_uri)
    return model_uri


if __name__ == "__main__":
    fire.Fire(get_last_model_uri)