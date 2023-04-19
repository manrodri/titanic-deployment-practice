from feature_engine.encoding import OneHotEncoder, RareLabelEncoder
from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from feature_engine.selection import DropFeatures
from feature_engine.transformation import LogTransformer
from feature_engine.wrappers import SklearnTransformerWrapper
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from regression_model.config.core import config
from regression_model.processing import features as pp

titanic_pipe = Pipeline(
    [
        # ===== IMPUTATION =====
        # impute categorical variables with string missing
        (
            "categorical_imputation",
            CategoricalImputer(
                imputation_method="missing",
                variables=config.model_config.categorical_vars_with_na_missing,
            ),
            # adding missing indicator to numerical variables
            (
                "missing_indicator",
                AddMissingIndicator(
                    variables=config.model_config.numerical_vars_with_na,
                )
            ),
            # impute numerical variables with the mean
            (
                "mean_imputation",
                MeanMedianImputer(
                    imputation_method="mean",
                    variables=config.model_config.numerical_vars_with_na,
                )
            ),
            # extract first letter from cabin
            (
                "extract_letter",
                pp.ExtractLetterTransformer(
                    variables=config.model_config.extract_first_letter_vars,
                )
            ),
            # == CATEGORICAL ENCODING
            (
                'rare_label_encoder',
                RareLabelEncoder(
                    tol=0.01, n_categories=1, variables=config.model_config.categorical_vars
                )
            ),
            # encode categorical variables using one hot encoding into k-1 variables
            ('categorical_encoder', OneHotEncoder(
                drop_last=True, variables=config.model_config.categorical_vars)),
            # scale
            ('scaler', StandardScaler()),

            ('Logit', LogisticRegression(
                C=config.model_config.C,
                random_state=config.model_config.random_state)
             ),
        )
    ]
)
