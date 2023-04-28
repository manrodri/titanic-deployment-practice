import typing as t
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __salary_prediction_model_version__ as _version
from regression_model.config.salary_core import DATASET_DIR, TRAINED_MODEL_DIR, config
from regression_model.processing.salary_validation import cast_numerical_variables_to_float


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    # copy dataframe
    transformed_dataframe = dataframe.copy()
    # cast numerical variables to float
    transformed_dataframe = cast_numerical_variables_to_float(input_data=transformed_dataframe)
    return transformed_dataframe


def load_pipeline(*, file_name: str) -> Pipeline:
    file_path = TRAINED_MODEL_DIR / file_name
    trained_model = joblib.load(filename=file_path)
    return trained_model


def remove_old_pipelines(*, files_to_keep: t.List[str]) -> None:
    """
    Remove old model pipelines.
    This is to ensure there is a simple one-to-one
    mapping between the package version and the model
    version to be imported and used by other applications.
    """
    do_not_delete = files_to_keep + ["__init__.py"]
    for model_file in TRAINED_MODEL_DIR.iterdir():
        if model_file.name not in do_not_delete:
            model_file.unlink()


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name
    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)
