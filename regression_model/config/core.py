from pathlib import Path
from typing import Dict, List, Optional, Sequence

from pydantic import BaseModel
from strictyaml import YAML, load

import regression_model

# Project Directories
PACKAGE_ROOT = Path(regression_model.__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent
CONFIG_FILE_PATH = PACKAGE_ROOT / "config.yml"
DATASET_DIR = PACKAGE_ROOT / "datasets"
TRAINED_MODEL_DIR = PACKAGE_ROOT / "trained_models"


class AppConfig(BaseModel):
    """
    Application-level config.
    """

    package_name: str
    training_data_file: str
    test_data_file: str
    pipeline_save_file: str


class ModelConfig(BaseModel):
    """
    All configuration relevant to model
    training and feature engineering.
    """

    features: List[str]
    test_size: float
    random_state: int
    C: float
    survived: str
    extract_first_letter_vars: List[str]
    variables_to_drop: List[str]
    categorical_vars: Sequence[str]
    categorical_vars_with_na_frequent: List[str]
    categorical_vars_with_na_missing: List[str]
    numerical_vars: Sequence[str]
    numerical_vars_with_na: List[str]
    numerical_vars_with_frequent: List[str]
    numerical_vars_to_cast_to_float: List[str]


def find_config_file() -> Path:
    """Locate the configuration file."""
    if CONFIG_FILE_PATH.is_file():
        return CONFIG_FILE_PATH
    else:
        raise FileNotFoundError("Could not find config.yml file.")


def fetch_config_from_yaml(cfg_path: Optional[Path] = None) -> YAML:
    """Fetch config from yaml file."""
    if not cfg_path:
        cfg_path = find_config_file()
    with open(cfg_path, "r") as f:
        cfg = load(f.read(), YAML())
        return cfg
    raise FileNotFoundError(f"Did not find config file at path: {cfg_path}")


def create_config(parsed_config: YAML = None) -> Config:
    """Create config object from yaml file."""
    if not parsed_config:
        parsed_config = fetch_config_from_yaml()
    return Config(
        app_config=AppConfig(**parsed_config["app_config"].data),
        model_config=ModelConfig(**parsed_config["model_config"].data),
    )


config = create_config()


