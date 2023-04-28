import pytest

from regression_model.config.salary_core import config
from regression_model.processing.salary_data_manager import load_dataset
from regression_model.processing.salary_validation import cast_numerical_variables_to_float
import numpy as np


@pytest.fixture
def salary_sample_input_data():
    # load test data
    test_data = load_dataset(file_name=config.app_config.test_data_file)
    return test_data


def test_cast_to_float(salary_sample_input_data):
    # Given
    print (salary_sample_input_data.columns)
    # assert type(salary_sample_input_data['YearsExperience'][0].dtype) == np.dtype["float32"]
    assert salary_sample_input_data['Salary'][0] == 122391.0

    # When
    X = cast_numerical_variables_to_float(input_data=salary_sample_input_data)

    # Then
    assert np.isclose(X['YearsExperience'][0], [10.3])
    assert X['Salary'][0] == 122391.0
    assert X['YearsExperience'].dtype == 'float32'
    assert X['Salary'].dtype == 'float32'

