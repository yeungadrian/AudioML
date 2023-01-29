# AudioML

## Tools used in this project
* [Poetry](https://python-poetry.org/docs/basic-usage/): Dependency management
* [hydra](https://hydra.cc/): Manage configuration 
* [pre-commit plugins](https://pre-commit.com/): Automate code reviewing formatting
* [pdoc](https://github.com/pdoc3/pdoc): Automatic documentation
* To be used in combination with https://github.com/yeungadrian/mlops
  * [MLFlow](https://mlflow.org/docs/latest/index.html) Experiment & Model Tracking
  * [Minio](https://docs.min.io) Simple S3 storage

## Project structure
```
.
├── config                      
│   ├── main.yaml                   # Main configuration file
├── notebooks                       # store notebooks
├── src                             # store source code
│   ├── __init__.py                 # make src a Python module 
│   └── train_model.py              # train model
├── tests                           # store tests
│   ├── __init__.py                 # make tests a Python module 
│   └── test_train_model.py         # test functions for train_model.py
├── .flake8                         # configuration for flake8 - a Python formatter tool
├── .gitignore                      # ignore files that cannot commit to Git
├── .pre-commit-config.yaml         # configurations for pre-commit
├── Makefile                        # store useful commands to set up the environment
├── pyproject.toml                  # dependencies for poetry
└── README.md                       # describe your project
```

## Set up the environment
1. Install [Poetry](https://python-poetry.org/docs/#installation)
2. Set up the environment:
```
make activate
make setup
```

## Install new packages
```
poetry add <package-name>
```

# Generate docss
```
make docs
```