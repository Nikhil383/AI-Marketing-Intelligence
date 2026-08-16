# src/mlops.py
import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

def calculate_psi(expected, actual, bins=10):
    """Calculate Population Stability Index (PSI)."""
    breakpoints = np.percentile(expected, np.linspace(0, 100, bins + 1))
    breakpoints[0], breakpoints[-1] = -np.inf, np.inf
    
    expected_counts = np.histogram(expected, bins=breakpoints)[0]
    actual_counts = np.histogram(actual, bins=breakpoints)[0]
    
    expected_pct = np.where(expected_counts == 0, 0.0001, expected_counts) / len(expected)
    actual_pct = np.where(actual_counts == 0, 0.0001, actual_counts) / len(actual)
    
    return np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))

def detect_data_drift(train_df, prod_df, feature):
    """Triggers flag if KS-Test p-value < 0.05 or PSI > 0.25."""
    stat, p_value = ks_2samp(train_df[feature].dropna(), prod_df[feature].dropna())
    psi_val = calculate_psi(train_df[feature].dropna(), prod_df[feature].dropna())
    
    is_drift = (p_value < 0.05) or (psi_val > 0.25)
    return {"drift_detected": is_drift, "p_value": p_value, "psi": psi_val}

class ThompsonSamplingABTester:
    """Beta-Binomial Thompson Sampling for dynamic traffic splitting."""
    def __init__(self):
        self.arms = {
            "Control_Human": {"alpha": 1, "beta": 1},
            "Treatment_AI": {"alpha": 1, "beta": 1}
        }
        
    def select_variant(self):
        samples = {arm: np.random.beta(d["alpha"], d["beta"]) for arm, d in self.arms.items()}
        return max(samples, key=samples.get)
        
    def update_reward(self, arm, converted: bool):
        if converted:
            self.arms[arm]["alpha"] += 1
        else:
            self.arms[arm]["beta"] += 1

if __name__ == "__main__":
    print("Executing standalone MLOps module checks...")
    train_data = pd.DataFrame({"Reach": np.random.normal(5000, 1000, 500)})
    prod_data = pd.DataFrame({"Reach": np.random.normal(5100, 1050, 100)})
    
    drift_result = detect_data_drift(train_data, prod_data, "Reach")
    print(f"Drift Analysis: {drift_result}")
    
    tester = ThompsonSamplingABTester()
    variant = tester.select_variant()
    print(f"Assigned A/B Test Variant: {variant}")