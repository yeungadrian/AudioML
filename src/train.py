import itertools
import logging
from typing import Any, List, Tuple

import hydra
import matplotlib.pyplot as plt
import mlflow
import numpy as np
from dotenv import load_dotenv
from .helpers import MLFlowUtils
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, Schema
from omegaconf import DictConfig, OmegaConf
from sklearn import metrics
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s:%(levelname)s:%(message)s",
    datefmt="%Y-%m-%d %I:%M:%S%p",
)

mlflowutils = MLFlowUtils()


def load_data() -> Any:
    """
    Load iris dataset

    Returns:
        Any

    """
    iris_dataset = load_iris()
    return iris_dataset


def process_data(
    iris_dataset: Any,
    test_size: float,
    random_state: int,
) -> Tuple[np.ndarray, ...]:
    """
    Generate test train split

    Args:
        iris_dataset (sklearn.utils._bunch.Bunch): Iris dataset
        test_size (float): Percentage of dataset to use as test set
        random_state (int): Random state to use for splits

    Returns:
        Tuple[np.ndarray, ...]: (X_train, X_test, y_train, y_test)

    """
    X_train, X_test, y_train, y_test = train_test_split(
        iris_dataset["data"],
        iris_dataset["target"],
        test_size=test_size,
        random_state=random_state,
    )
    return X_train, X_test, y_train, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray) -> Any:
    """
    Train sklearn decision tree classifier

    Args:
        X_train (np.ndarray): X_train
        y_train (np.ndarray): y_train

    Returns:
        Any: Fitted model

    """
    model = DecisionTreeClassifier()
    model.fit(X_train, y_train)
    return model


def log_model(
    model: Any,
    model_path: str,
) -> Tuple[str, str]:
    """
    Log model params, model and model signature (inputs / outputs) to mlflow

    Args:
        model (Any): Sklearn Decision tree model
        model_path (str): Artifact path in mlflow

    Returns:
        Tuple of model_uri and run_id

    """
    input_schema = Schema(
        [
            ColSpec("double", "sepal length (cm)"),
            ColSpec("double", "sepal width (cm)"),
            ColSpec("double", "petal length (cm)"),
            ColSpec("double", "petal width (cm)"),
        ]
    )
    output_schema = Schema([ColSpec("long")])
    signature = ModelSignature(inputs=input_schema, outputs=output_schema)
    mlflow.log_params(model.get_params())
    result = mlflow.sklearn.log_model(model, model_path, signature=signature)
    return result.model_uri, result.run_id


def get_preds(model: Any, X_test: np.ndarray) -> np.ndarray:
    """
    Get predictions from sklearn model

    Args:
        model (Any): sklearn model
        X_test (np.ndarray): data

    Returns:
        np.ndarray: Probability predictions

    """
    probs = model.predict_proba(X_test)
    y_pred = probs[:, 1]
    return y_pred


def plot_confusion_matrix(
    cm: np.ndarray,
    classes: List,
    normalize: bool = False,
    title: str = "Confusion matrix",
) -> plt.figure:
    """
    Generate multiclass confusion matrix plot

    Args:
        cm (np.ndarray): Confusion matrix results
        classes (List): List of all classes
        normalize (bool=False): Show normalised results
        title (str="Confusionmatrix"): Plot title

    Returns:
        plt.figure: confusion matrix plot

    """
    cmap = plt.cm.Blues
    fig = plt.figure()
    plt.imshow(cm, interpolation="nearest", cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    if normalize:
        cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    thresh = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(
            j,
            i,
            cm[i, j],
            horizontalalignment="center",
            color="white" if cm[i, j] > thresh else "black",
        )
    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return fig


def log_model_validation(
    model: Any, X_test: np.ndarray, y_test: np.ndarray
) -> float:
    """
    Log model validation metrics to mlflow
    - Training curves
    - AUC
    - Prediction distributions
    - Confusion matrices

    Args:
        model (Any): model
        X_test (np.ndarray): X_test
        y_test (np.ndarray): y_test

    Returns:
        float: balanced accuracy score

    """
    y_pred = get_preds(model, X_test)
    bacc_score = metrics.balanced_accuracy_score(y_test, y_pred)
    mlflow.log_metric("balanced_accuracy_score", bacc_score)
    cm = metrics.confusion_matrix(y_test, y_pred)
    cm_png = plot_confusion_matrix(cm, classes=list(np.unique(y_test)))
    mlflow.log_figure(cm_png, "confusionmatrix.png")
    return bacc_score


@hydra.main(config_path="../config", config_name="main")
def main(config: DictConfig) -> None:
    load_dotenv()
    tags = OmegaConf.to_container(config.mlflow.tags)
    mlflowutils.start_mlflow_run(config.mlflow.experiment_name, tags=tags)
    iris_dataset = load_data()
    X_train, X_test, y_train, y_test = process_data(
        iris_dataset, config.mlflow.test_size, config.mlflow.random_state
    )
    model = train_model(X_train, y_train)
    model_uri, run_id = log_model(model, config.mlflow.model_file_path)
    bacc_score = log_model_validation(model, X_test, y_test)
    mlflow.end_run()
    model_version = mlflowutils.register_model(
        bacc_score,
        model_uri,
        run_id,
        config.mlflow.threshold,
        config.mlflow.model_registry_name,
        tags,
    )
    if model_version is not None:
        mlflowutils.promote_model(
            model_version, config.mlflow.model_registry_name
        )


if __name__ == "__main__":
    main()
