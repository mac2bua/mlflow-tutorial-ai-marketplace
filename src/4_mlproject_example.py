"""
Downloads the MovieLens dataset, ETLs it into Parquet, trains an
ALS model, and uses the ALS model to train a Keras neural network.
See README.rst for more details.
"""

import click
import os
from dotenv import load_dotenv

import mlflow
from mlflow.utils import mlflow_tags
from mlflow.entities import RunStatus
from mlflow.utils.logging_utils import eprint

from mlflow.tracking.fluent import _get_experiment_id

load_dotenv()

experiment_name = "pipeline-example"
mlflow.set_experiment(experiment_name)

def _run(entrypoint: str, parameters: dict, project_dir: str = '../mlflow-project/'):
    """Launches an entry point by providing the given parameters."""

    print("Launching new run for entrypoint=%s and parameters=%s" % (entrypoint, parameters))
    submitted_run = mlflow.run(project_dir, entrypoint, parameters=parameters, storage_dir='../data')

    client = mlflow.tracking.MlflowClient()
    return client.get_run(submitted_run.run_id)


@click.command()
@click.option("--als-max-iter", default=10, type=int)
@click.option("--keras-hidden-units", default=20, type=int)
@click.option("--max-row-limit", default=100000, type=int)
def mlflow_pipeline(als_max_iter, keras_hidden_units, max_row_limit):
    # Note: The entrypoint names are defined in MLproject. The artifact directories
    # are documented by each step's .py file.
    with mlflow.start_run() as active_run:
        os.environ["SPARK_CONF_DIR"] = os.path.abspath(".")

        load_raw_data_run = _run("load_raw_data", {})
        ratings_csv_uri = os.path.join(load_raw_data_run.info.artifact_uri, "ratings-csv-dir")

        etl_data_run = _run("etl_data", {"ratings_csv": ratings_csv_uri, "max_row_limit": max_row_limit})
        ratings_parquet_uri = os.path.join(etl_data_run.info.artifact_uri, "ratings-parquet-dir")

        als_run = _run("als", {"ratings_data": ratings_parquet_uri, "max_iter": str(als_max_iter)})
        als_model_uri = os.path.join(als_run.info.artifact_uri, "als-model")

        keras_params = {
            "ratings_data": ratings_parquet_uri,
            "als_model_uri": als_model_uri,
            "hidden_units": keras_hidden_units,
        }
        _run("train_keras", keras_params)


if __name__ == "__main__":
    mlflow_pipeline()
