# src/train.py

import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from preprocess import build_preprocessor

from config import (
    DATA_PATH,
    MODEL_PATH,
    CAT_FEATURES,
    NUM_FEATURES,
    TEST_SIZE,
    RANDOM_STATE,
    N_ESTIMATORS,
    LEARNING_RATE
)


def train_model():

    print("Loading dataset...")

    df = pd.read_csv(DATA_PATH)

    df = df.dropna(
        subset=["Performance_Score"]
    )

    features = CAT_FEATURES + NUM_FEATURES

    X = df[features]
    y = df["Performance_Score"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )

    # -----------------------------
    # Train model
    # -----------------------------

    model_pipeline = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor()
            ),
            (
                "regressor",
                LGBMRegressor(
                    n_estimators=N_ESTIMATORS,
                    learning_rate=LEARNING_RATE,
                    random_state=RANDOM_STATE,
                    verbosity=-1
                )
            )
        ]
    )

    print("Training LightGBM...")

    model_pipeline.fit(
        X_train,
        y_train
    )

    # -----------------------------
    # Evaluation
    # -----------------------------

    predictions = model_pipeline.predict(X_test)

    r2 = r2_score(
        y_test,
        predictions
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    print("\nModel Evaluation")
    print("----------------")
    print(f"R2   : {r2:.4f}")
    print(f"MAE  : {mae:.2f}")
    print(f"RMSE : {rmse:.2f}")

    # -----------------------------
    # Historical content performance
    # -----------------------------

    content_performance = (
        df.groupby("Content_Type")
        ["Performance_Score"]
        .agg(
            Average_Score="mean",
            Number_of_Records="count"
        )
        .reset_index()
        .sort_values(
            "Average_Score",
            ascending=False
        )
    )

    content_performance[
        "Average_Score"
    ] = content_performance[
        "Average_Score"
    ].round(2)

    # -----------------------------
    # Save recommendation table
    # -----------------------------

    os.makedirs("data", exist_ok=True)

    recommendation_csv = (
        "data/content_performance.csv"
    )

    content_performance.to_csv(
        recommendation_csv,
        index=False
    )

    # -----------------------------
    # Save recommendation image
    # -----------------------------

    plt.figure(
        figsize=(10, 6)
    )

    plt.bar(
        content_performance["Content_Type"],
        content_performance["Average_Score"]
    )

    plt.xlabel(
        "Content Type"
    )

    plt.ylabel(
        "Average Performance Score"
    )

    plt.title(
        "Historical Content Performance"
    )

    plt.xticks(
        rotation=25
    )

    plt.tight_layout()

    recommendation_image = (
        "data/content_performance.png"
    )

    plt.savefig(
        recommendation_image,
        dpi=150
    )

    plt.close()

    # -----------------------------
    # Save model artifacts
    # -----------------------------

    artifacts = {
        "model": model_pipeline,

        "historical_content_performance":
            content_performance,

        "metrics": {
            "r2": r2,
            "mae": mae,
            "rmse": rmse
        }
    }

    joblib.dump(
        artifacts,
        MODEL_PATH
    )

    print(
        f"\nModel saved: {MODEL_PATH}"
    )

    print(
        f"Recommendation table saved: "
        f"{recommendation_csv}"
    )

    print(
        f"Recommendation image saved: "
        f"{recommendation_image}"
    )


if __name__ == "__main__":
    train_model()