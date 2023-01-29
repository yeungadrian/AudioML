from src.training_pipeline import *
from hydra import initialize, compose

def test_loading_data():
    with initialize(version_base=None, config_path="../config"):
        config = compose(config_name="main")
    iris_dataset = load_data()
    X_train, X_test, y_train, y_test = process_data(
        iris_dataset, config.mlflow.test_size, config.mlflow.random_state
    )
    assert X_train.shape[0] == 112
    assert X_train.shape[1] == 4