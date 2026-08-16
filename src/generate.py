# src/generate.py

import os

import numpy as np
import pandas as pd

from config import (
    DATA_PATH,
    INDUSTRIES,
    PLATFORMS,
    CONTENT_TYPES,
    TOPICS
)


def generate_dirty_dataset(n=600):

    np.random.seed(42)

    # Random timestamps
    random_dates = pd.to_datetime(
        np.random.choice(
            pd.date_range(
                "2024-01-01",
                "2026-12-31",
                freq="h"
            ),
            n
        )
    )

    # Random categorical features
    industry = np.random.choice(
        INDUSTRIES,
        n
    )

    platform = np.random.choice(
        PLATFORMS,
        n,
        p=[0.35, 0.25, 0.20, 0.10, 0.10]
    )

    content_type = np.random.choice(
        CONTENT_TYPES,
        n
    )

    topic = np.random.choice(
        TOPICS,
        n
    )

    df = pd.DataFrame(
        {
            "Client": [
                f"Client_{i:02d}"
                for i in np.random.randint(1, 16, n)
            ],
            "Industry": industry,
            "Platform": platform,
            "Content_Type": content_type,
            "Content_Topic": topic,
            "Posting_Timestamp": random_dates
        }
    )

    # Time features
    df["Posting_Day"] = (
        df["Posting_Timestamp"].dt.day_name()
    )

    df["Posting_Hour"] = (
        df["Posting_Timestamp"].dt.hour
    )

    # Campaign inputs
    df["Reach"] = np.random.randint(
        1500,
        15000,
        n
    )

    df["Ad_Spend"] = np.round(
        np.random.uniform(50, 500, n),
        2
    )

    # ----------------------------------
    # Synthetic performance relationships
    # ----------------------------------

    # Platform effect
    platform_boost = np.select(
        [
            platform == "Instagram",
            platform == "LinkedIn",
            platform == "YouTube",
            platform == "Facebook",
            platform == "X"
        ],
        [
            15,
            10,
            12,
            7,
            5
        ],
        default=5
    )

    # Content type effect
    content_boost = np.select(
        [
            content_type == "Reel",
            content_type == "Case Study",
            content_type == "Carousel",
            content_type == "Infographic",
            content_type == "Blog Post"
        ],
        [
            20,
            14,
            10,
            8,
            5
        ],
        default=5
    )

    # Topic effect
    topic_boost = np.select(
        [
            topic == "Education",
            topic == "Customer Success",
            topic == "Trends",
            topic == "Behind the Scenes",
            topic == "Promo"
        ],
        [
            10,
            12,
            8,
            6,
            3
        ],
        default=3
    )

    # Posting time effect
    hour_boost = np.select(
        [
            (df["Posting_Hour"] >= 18)
            & (df["Posting_Hour"] <= 21),

            (df["Posting_Hour"] >= 10)
            & (df["Posting_Hour"] <= 13)
        ],
        [
            8,
            4
        ],
        default=0
    )

    # Reach contribution
    reach_effect = (
        np.log1p(df["Reach"]) * 1.5
    )

    # Ad spend contribution
    spend_effect = (
        np.log1p(df["Ad_Spend"]) * 1.5
    )

    # Random noise
    noise = np.random.normal(
        0,
        3,
        n
    )

    # Final performance score
    df["Performance_Score"] = np.clip(
        35
        + platform_boost
        + content_boost
        + topic_boost
        + hour_boost
        + reach_effect
        + spend_effect
        + noise,
        15,
        98
    ).round().astype(int)

    # ----------------------------------
    # Dirty data / edge cases
    # ----------------------------------

    # Missing topics
    missing_topic_idx = np.random.choice(
        n,
        int(n * 0.10),
        replace=False
    )

    df.loc[
        missing_topic_idx,
        "Content_Topic"
    ] = np.nan

    # Missing ad spend
    missing_spend_idx = np.random.choice(
        n,
        int(n * 0.05),
        replace=False
    )

    df.loc[
        missing_spend_idx,
        "Ad_Spend"
    ] = np.nan

    # Reach outliers
    outlier_idx = np.random.choice(
        n,
        5,
        replace=False
    )

    df.loc[
        outlier_idx,
        "Reach"
    ] = 9999999

    # Create data directory
    os.makedirs(
        "data",
        exist_ok=True
    )

    df.to_csv(
        DATA_PATH,
        index=False
    )

    print(
        f"Generated {n} marketing records at {DATA_PATH}"
    )


if __name__ == "__main__":
    generate_dirty_dataset()