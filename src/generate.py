# src/generate.py
import pandas as pd
import numpy as np
import os
from config import *

def generate_dirty_dataset(n=600):
    np.random.seed(42)
    random_dates = pd.to_datetime(np.random.choice(pd.date_range('2024-01-01', '2026-12-31', freq='h'), n))
    
    industry = np.random.choice(INDUSTRIES, n)
    platform = np.random.choice(PLATFORMS, n, p=[0.35, 0.25, 0.20, 0.10, 0.10])
    content_type = np.random.choice(CONTENT_TYPES, n)
    topic = np.random.choice(TOPICS, n)
    
    df = pd.DataFrame({
        "Client": [f"Client_{i:02d}" for i in np.random.randint(1, 16, n)],
        "Industry": industry,
        "Platform": platform,
        "Content_Type": content_type,
        "Content_Topic": topic,
        "Posting_Timestamp": random_dates
    })
    
    df["Posting_Day"] = df["Posting_Timestamp"].dt.day_name()
    df["Posting_Hour"] = df["Posting_Timestamp"].dt.hour
    df["Reach"] = np.random.randint(1500, 15000, n)
    df["Ad_Spend"] = np.round(np.random.uniform(50, 500, n), 2)
    
    # Tie target score to features so LightGBM can learn a real relationship
    platform_boost = np.select([platform == 'Instagram', platform == 'LinkedIn'], [15, 10], default=5)
    format_boost = np.select([content_type == 'Reel', content_type == 'Case Study'], [20, 12], default=5)
    noise = np.random.normal(0, 3, n)
    
    df["Performance_Score"] = np.clip(40 + platform_boost + format_boost + (np.log1p(df["Reach"]) * 1.5) + noise, 15, 98).astype(int)

    # Inject edge cases (Nulls & Outliers)
    df.loc[np.random.choice(n, int(n * 0.1), replace=False), "Content_Topic"] = np.nan
    df.loc[np.random.choice(n, int(n * 0.05), replace=False), "Ad_Spend"] = np.nan
    df.loc[np.random.choice(n, 5, replace=False), "Reach"] = 9999999

    os.makedirs("data", exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Generated clean correlated dataset at {DATA_PATH}")

if __name__ == "__main__":
    generate_dirty_dataset()