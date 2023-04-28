import math

from regression_model.predict import make_prediction
from tests.conf_tests import sample_input_data


def test_make_single_prediction():
    # Given
    expected_first_prediction = 0.0
    expected_number_of_predictions = 1449

    # When
    result = make_prediction(input_data=sample_input_data)

    # Then
    preds = result.get("predictions")
    assert isinstance(preds, list)
    assert len(preds) == expected_number_of_predictions
    assert result.get("errors") is None
    assert math.isclose(preds[0], expected_first_prediction, rel_tol=1e-3)