import pytest

from regression_model.config.core import config
from regression_model.processing.data_manager import load_dataset


@pytest.fixture()
def sample_input_data():
    # load test data
    test_data = load_dataset(file_name=config.app_config.test_data_file)
    return test_data


@pytest.fixture()
def sample_preproccessed_data():
    preprocessed_data = load_dataset(
        file_name=config.app_config.preproccessed_data_file
    )
    return preprocessed_data
