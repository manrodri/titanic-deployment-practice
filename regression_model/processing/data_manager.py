import typing as t
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from regression_model import __version__ as _version
from regression_model.config.core import DATASET_DIR, TRAINED_MODEL_DIR, config
from regression_model.processing.validation import (
    cast_numerical_variables_to_float,
    drop_unnecessary_variables,
    get_first_cabin,
    get_title,
)


def load_dataset(*, file_name: str) -> pd.DataFrame:
    dataframe = pd.read_csv(Path(f"{DATASET_DIR}/{file_name}"))
    # copy dataframe
    transformed_dataframe = dataframe.copy()
    # replace question marks with NaN
    transformed_dataframe = transformed_dataframe.replace("?", np.NaN)
    # retain only the first cabin if more than
    # 1 are available per passenger
    transformed_dataframe = get_first_cabin(input_data=transformed_dataframe)
    # extract the title from the name
    transformed_dataframe = get_title(input_data=transformed_dataframe)
    # cast numerical variables to float
    cast_numerical_variables_to_float(input_data=transformed_dataframe)
    # drop unnecessary columns
    drop_unnecessary_variables(input_data=transformed_dataframe)

    return transformed_dataframe


def save_pipeline(*, pipeline_to_persist: Pipeline) -> None:
    """Persist the pipeline.
    Saves the versioned model, and overwrites any previous
    saved models. This ensures that when the package is
    published, there is only one trained model that can be
    called, and we know exactly how it was built.
    """

    # Prepare versioned save file name
    save_file_name = f"{config.app_config.pipeline_save_file}{_version}.pkl"
    save_path = TRAINED_MODEL_DIR / save_file_name

    remove_old_pipelines(files_to_keep=[save_file_name])
    joblib.dump(pipeline_to_persist, save_path)


def load_pipeline(*, file_name: str) -> Pipeline:
    """Load a persisted pipeline."""

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
