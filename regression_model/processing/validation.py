import re
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from pydantic import BaseModel, ValidationError

from regression_model.config.core import config


def drop_na_inputs(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Check model inputs for na values and filter."""
    validated_data = input_data.copy()
    new_vars_with_na = [
        var
        for var in config.model_config.features
        if var
        not in config.model_config.categorical_vars_with_na_frequent
        + config.model_config.categorical_vars_with_na_missing
        + config.model_config.numerical_vars_with_na
        and validated_data[var].isnull().sum() > 0
    ]
    validated_data.dropna(subset=new_vars_with_na, inplace=True)

    return validated_data


# def validate_inputs(*, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[dict]]:
#     """Check model inputs for unprocessable values."""
#
#     # convert syntax error field names (beginning with numbers)
#     input_data.rename(columns=config.model_config.variables_to_rename, inplace=True)
#     input_data["MSSubClass"] = input_data["MSSubClass"].astype("O")
#     relevant_data = input_data[config.model_config.features].copy()
#     validated_data = drop_na_inputs(input_data=relevant_data)
#     errors = None
#
#     try:
#         # replace numpy nans so that pydantic can validate
#         MultipleHouseDataInputs(
#             inputs=validated_data.replace({np.nan: None}).to_dict(orient="records")
#         )
#     except ValidationError as error:
#         errors = error.json()
#
#     return validated_data, errors

def drop_unnecessary_variables(*, input_data: pd.DataFrame) -> pd.DataFrame:
    """Drop unnecessary variables."""
    validated_data = input_data.copy()
    validated_data.drop(config.model_config.variables_to_drop, axis=1, inplace=True)

    return validated_data


def cast_numerical_variables_to_float(input_data: pd.DataFrame) -> pd.DataFrame:
    """Cast numerical variables to float."""
    validated_data = input_data.copy()
    for var in config.model_config.numerical_vars_to_cast_to_float:
        validated_data[var] = validated_data[var].astype(float)

    return validated_data


def get_first_cabin(input_data: pd.DataFrame) -> pd.DataFrame:
    """Extract first letter of cabin."""
    validated_data = input_data.copy()

    def func(passenger):
        try:
            return passenger.split()[0]
        except Exception:
            return np.nan

    validated_data["cabin"] = validated_data["cabin"].apply(func)

    return validated_data


def get_title(input_data: pd.DataFrame) -> pd.DataFrame:
    """Extract title from name."""
    validated_data = input_data.copy()

    # extracts the title (Mr, Ms, etc) from the name variable

    def func(passenger):
        line = passenger
        if re.search("Mrs", line):
            return "Mrs"
        elif re.search("Mr", line):
            return "Mr"
        elif re.search("Miss", line):
            return "Miss"
        elif re.search("Master", line):
            return "Master"
        else:
            return "Other"

    validated_data["title"] = validated_data["name"].apply(func)

    return validated_data


class PassengerDataInputSchema(BaseModel):
    pclass: Optional[int]
    survived: Optional[int]
    name: Optional[str]
    sex: Optional[str]
    age: Optional[str]
    sibsp: Optional[int]
    parch: Optional[int]
    ticket: Optional[str]
    fare: Optional[str]
    cabin: Optional[str]
    embarked: Optional[str]
    boat: Optional[str]
    body: Optional[str]
    home_dest: Optional[str]


class MultiplePassengerDataInputs(BaseModel):
    inputs: List[PassengerDataInputSchema]
