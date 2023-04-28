import typing as t
import numpy as np
import pandas as pd

from regression_model import __salary_prediction_model_version__ as _version
from regression_model.config.salary_core import config
from regression_model.processing.salary_data_manager import load_pipeline
from regression_model.processing.salary_validation import validate_inputs

pipeline = f"{config.app_config.pipeline_save_file}{_version}.pkl"
_salary_pipeline = load_pipeline(file_name=pipeline)


def make_prediction(
        *,
        input_data: t.Union[pd.DataFrame, dict],
) -> dict:
    data = pd.DataFrame(input_data)
    validated_data, errors = validate_inputs(input_data=data)
    results = {"predictions": None, "version": _version, "errors": errors}

    if not errors:
        predictions = _salary_pipeline.predict(
            X=validated_data[config.model_config.features]
        )
        results = {
            "predictions": [np.exp(pred) for pred in predictions],  # type: ignore
            "version": _version,
            "errors": errors,
        }

    return results
