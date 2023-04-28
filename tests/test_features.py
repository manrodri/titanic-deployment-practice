from regression_model.config.core import config
from regression_model.processing.features import ExtractLetterTransformer
from regression_model.processing.validation import get_title


def test_get_title(sample_input_data):
    # Given
    assert sample_input_data["name"][0] == "Allen, Miss. Elisabeth Walton"
    assert sample_input_data["name"][1] == "Allison, Master. Hudson Trevor"

    # When
    X = get_title(input_data=sample_input_data)

    # Then
    assert X["title"].nunique() <= 5
    assert X["title"][0] == "Miss"
    assert X["title"][1] == "Master"
    # assert X["title"][10] == 'Other'


def test_extract_letter_transformer(sample_preproccessed_data):
    # Given
    transformer = ExtractLetterTransformer(
        variables=config.model_config.extract_first_letter_vars
    )
    assert sample_preproccessed_data["cabin"][0] == "B5"
    assert sample_preproccessed_data["cabin"][1] == "C22"

    # When
    X = transformer.fit_transform(sample_preproccessed_data)

    # Then
    assert X["cabin"][0] == "B"
    assert X["cabin"][1] == "C"
    assert X["cabin"].nunique() <= 8
    assert X["cabin"].isnull().sum() == 1014
