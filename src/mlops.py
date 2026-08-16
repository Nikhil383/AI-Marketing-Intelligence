# src/mlops.py

import numpy as np
import pandas as pd

from scipy.stats import ks_2samp


# ---------------------------------
# PSI
# ---------------------------------

def calculate_psi(
    expected,
    actual,
    bins=10
):
    """
    Calculate Population Stability Index.
    """

    expected = np.asarray(
        expected,
        dtype=float
    )

    actual = np.asarray(
        actual,
        dtype=float
    )

    expected = expected[
        np.isfinite(expected)
    ]

    actual = actual[
        np.isfinite(actual)
    ]

    if len(expected) == 0 or len(actual) == 0:
        return np.nan

    breakpoints = np.percentile(
        expected,
        np.linspace(
            0,
            100,
            bins + 1
        )
    )

    breakpoints = np.unique(
        breakpoints
    )

    if len(breakpoints) < 3:

        return 0.0

    breakpoints[0] = -np.inf
    breakpoints[-1] = np.inf

    expected_counts = np.histogram(
        expected,
        bins=breakpoints
    )[0]

    actual_counts = np.histogram(
        actual,
        bins=breakpoints
    )[0]

    expected_pct = (
        expected_counts /
        len(expected)
    )

    actual_pct = (
        actual_counts /
        len(actual)
    )

    # Avoid log(0)
    expected_pct = np.where(
        expected_pct == 0,
        0.0001,
        expected_pct
    )

    actual_pct = np.where(
        actual_pct == 0,
        0.0001,
        actual_pct
    )

    psi = np.sum(
        (
            actual_pct - expected_pct
        )
        *
        np.log(
            actual_pct /
            expected_pct
        )
    )

    return float(psi)


# ---------------------------------
# Numerical feature drift
# ---------------------------------

def detect_data_drift(
    train_df,
    prod_df,
    feature
):
    """
    Detect drift using:
    1. Kolmogorov-Smirnov test
    2. Population Stability Index
    """

    train_values = pd.to_numeric(
        train_df[feature],
        errors="coerce"
    ).dropna()

    prod_values = pd.to_numeric(
        prod_df[feature],
        errors="coerce"
    ).dropna()

    if (
        len(train_values) == 0
        or len(prod_values) == 0
    ):

        return {
            "drift_detected": False,
            "p_value": None,
            "psi": None
        }

    statistic, p_value = ks_2samp(
        train_values,
        prod_values
    )

    psi_value = calculate_psi(
        train_values,
        prod_values
    )

    drift_detected = (
        p_value < 0.05
        or psi_value > 0.25
    )

    return {
        "drift_detected": bool(
            drift_detected
        ),
        "ks_statistic": float(
            statistic
        ),
        "p_value": float(
            p_value
        ),
        "psi": float(
            psi_value
        )
    }


# ---------------------------------
# Categorical drift
# ---------------------------------

def detect_categorical_drift(
    train_df,
    prod_df,
    feature,
    threshold=0.20
):
    """
    Detect categorical distribution changes.

    A feature is considered to have drift
    if the largest absolute category-frequency
    change exceeds the threshold.
    """

    train_distribution = (
        train_df[feature]
        .fillna("Unknown")
        .value_counts(
            normalize=True
        )
    )

    prod_distribution = (
        prod_df[feature]
        .fillna("Unknown")
        .value_counts(
            normalize=True
        )
    )

    categories = set(
        train_distribution.index
    ).union(
        prod_distribution.index
    )

    max_difference = 0.0

    for category in categories:

        train_pct = train_distribution.get(
            category,
            0
        )

        prod_pct = prod_distribution.get(
            category,
            0
        )

        difference = abs(
            train_pct - prod_pct
        )

        max_difference = max(
            max_difference,
            difference
        )

    return {
        "drift_detected":
            max_difference > threshold,

        "max_distribution_change":
            float(max_difference)
    }


# ---------------------------------
# Thompson Sampling A/B testing
# ---------------------------------

class ThompsonSamplingABTester:

    """
    Beta-Binomial Thompson Sampling
    for Human vs AI recommendation comparison.
    """

    def __init__(self):

        self.arms = {
            "Control_Human": {
                "alpha": 1,
                "beta": 1
            },

            "Treatment_AI": {
                "alpha": 1,
                "beta": 1
            }
        }

    def select_variant(self):

        samples = {
            arm: np.random.beta(
                data["alpha"],
                data["beta"]
            )

            for arm, data
            in self.arms.items()
        }

        return max(
            samples,
            key=samples.get
        )

    def update_reward(
        self,
        arm,
        converted
    ):

        if arm not in self.arms:

            raise ValueError(
                f"Unknown arm: {arm}"
            )

        if converted:

            self.arms[
                arm
            ]["alpha"] += 1

        else:

            self.arms[
                arm
            ]["beta"] += 1


# ---------------------------------
# Standalone MLOps test
# ---------------------------------

if __name__ == "__main__":

    print(
        "Executing MLOps module checks..."
    )

    # Numerical drift example
    train_data = pd.DataFrame(
        {
            "Reach":
                np.random.normal(
                    5000,
                    1000,
                    500
                )
        }
    )

    production_data = pd.DataFrame(
        {
            "Reach":
                np.random.normal(
                    5100,
                    1050,
                    100
                )
        }
    )

    drift_result = detect_data_drift(
        train_data,
        production_data,
        "Reach"
    )

    print(
        "\nNumerical Drift:"
    )

    print(
        drift_result
    )

    # A/B test
    tester = (
        ThompsonSamplingABTester()
    )

    variant = (
        tester.select_variant()
    )

    print(
        f"\nSelected A/B variant: {variant}"
    )

    tester.update_reward(
        variant,
        converted=True
    )

    print(
        "Updated A/B test state:"
    )

    print(
        tester.arms
    )