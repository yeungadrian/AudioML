import logging
from typing import Any, Dict

import mlflow
from mlflow.entities.lifecycle_stage import LifecycleStage
from mlflow.exceptions import MlflowException
from mlflow.tracking import MlflowClient


class MLFlowUtils:
    def create_experiment(
        self, mlflow_client: mlflow.MlflowClient, experiment_name: str
    ) -> str:
        """
        Creates experiment in MLFlow if it doesn't exist

        Args:
            mlflow_client mlflow.MlflowClient(): mlflow client
            experiment_name (str): Experiennt Name

        Returns:
            str: Experiment ID

        """
        try:
            experiment_id = mlflow_client.create_experiment(experiment_name)
            logging.info(
                f"Experiment {experiment_name} created with id {experiment_id}"
            )
        except MlflowException as error:
            experiment = mlflow_client.get_experiment_by_name(experiment_name)
            if experiment.lifecycle_stage == LifecycleStage.DELETED:
                logging.error(f"Experiment {experiment_name} already DELETED")
                raise
            experiment_id = experiment.experiment_id
            logging.info(
                f"Experiment {experiment_name} exists with id {experiment_id}"
            )
        return experiment_id

    def start_mlflow_run(
        self, experiment_name: str, run_name: str = None, tags: dict = None
    ) -> mlflow.ActiveRun:
        """
        Start an mlflow run under specified experiment.
        Will generate experiment if it doesn't exist.

        Args:
            experiment_name (str): Experiment name in MLFlow
            run_name (str=None): Optional run name

        Returns:
            mlflow.ActiveRun

        """
        mlflow_client = MlflowClient()
        experiment_id = self.create_experiment(mlflow_client, experiment_name)
        run = mlflow.start_run(
            experiment_id=experiment_id, run_name=run_name, tags=tags
        )
        return run

    def register_model_name(
        self,
        mlflow_client: mlflow.MlflowClient,
        model_name: str,
        description: str = None,
    ) -> None:
        """
        Create new registered model if it doesn't exist

        Args:
            mlflow_client (mlflow.MlflowClient): mlflow client
            model_name (str): Name of registered model
            description (str=None): Model description

        Returns:
            None

        """
        try:
            mlflow_client.create_registered_model(model_name, description)
            logging.info(f"Registered model {model_name} created")
        except MlflowException as create_error:
            logging.info(f"Registered model {model_name} already exists")

        return None

    def register_model(
        self,
        model_score: float,
        model_uri: str,
        run_id: str,
        threshold: float,
        model_registry_name: str,
        tags: Dict = None,
    ) -> Any:
        """
        Register model from experiment run as new version

        Args:
            model_score (float): validation model score
            model_uri (str): model uri of experiment run
            run_id (str): experiment run id that generated model
            threshold (float): threshold score to register model
            model_registry_name (str): model name
            tags (Dict): tags to attach to model

        Returns:
            Any: registered model version as str or nothing

        """
        if model_score > threshold:
            logging.info(
                f"Model being registered. Model score of: {model_score}"
            )
            mlflow_client = MlflowClient()
            self.register_model_name(mlflow_client, model_registry_name)
            result = mlflow_client.create_model_version(
                model_registry_name,
                model_uri,
                run_id=run_id,
                tags=tags,
            )
            return result.version
        else:
            logging.warning(
                f"Model was not registered. Model score of: {model_score}"
            )
            return None

    def promote_model(
        self, model_version: str, model_registry_name: str
    ) -> None:
        """
        Promote model to "production" stage on MLFlow

        Args:
            model_version (str): model version to promote
            model_registry_name (str): model name

        Returns:
            None

        """
        mlflow_client = MlflowClient()
        mlflow_client.transition_model_version_stage(
            name=model_registry_name,
            version=model_version,
            stage="Production",
            archive_existing_versions=True,
        )
        return None
