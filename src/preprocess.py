# src/preprocess.py

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import OneHotEncoder, RobustScaler

from config import CAT_FEATURES, NUM_FEATURES


def build_preprocessor():

    # Categorical features
    cat_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="constant",
                    fill_value="Unknown"
                )
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False
                )
            )
        ]
    )

    # Numerical features
    num_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                KNNImputer(n_neighbors=5)
            ),
            (
                "scaler",
                RobustScaler()
            )
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "cat",
                cat_pipeline,
                CAT_FEATURES
            ),
            (
                "num",
                num_pipeline,
                NUM_FEATURES
            )
        ],
        remainder="drop"
    )

    return preprocessor