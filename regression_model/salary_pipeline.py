from feature_engine.imputation import AddMissingIndicator, CategoricalImputer, MeanMedianImputer
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline


salary_pipeline = Pipeline(
    [
        (
            "linear_regression",
            LinearRegression(
                fit_intercept=True,
            )
        )
    ]
)
