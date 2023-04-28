from typing import Tuple, Optional, List

import numpy as np
import pandas as pd
from pydantic import ValidationError, BaseModel

from regression_model.config.salary_core import config


def cast_numerical_variables_to_float(input_data: pd.DataFrame) -> pd.DataFrame:
    """Cast numerical variables to float."""
    validated_data = input_data.copy()
    for var in config.model_config.numerical_vars_to_cast_to_float:
        validated_data[var] = validated_data[var].astype('float32')

    return validated_data


def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
    """Check model inputs for unprocessable values."""

    # convert syntax error field names (beginning with numbers)
    input_data = cast_numerical_variables_to_float(input_data=input_data)
    validated_data = input_data[config.model_config.features].copy()
    errors = None

    try:
        # replace numpy nans so that pydantic can validate
        MultipleEmployeeDataInputs(
            inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
        )
    except ValidationError as error:
        errors = error.json()

    return validated_data, errors


class EmployeeDataInput(BaseModel):
    YearExperience: float


class MultipleEmployeeDataInputs(BaseModel):
    inputs: List[EmployeeDataInput]
