# src/train.py
import pandas as pd
import numpy as np
import joblib
from lightgbm import LGBMRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from preprocess import build_preprocessor
from config import DATA_PATH, MODEL_PATH, CAT_FEATURES, NUM_FEATURES

def train_model():
    df = pd.read_csv(DATA_PATH).dropna(subset=["Performance_Score"])
    
    X = df[CAT_FEATURES + NUM_FEATURES]
    y = df["Performance_Score"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Integrated LightGBM into the scikit-learn pipeline
    model_pipeline = Pipeline(steps=[
        ('preprocessor', build_preprocessor()),
        ('regressor', LGBMRegressor(n_estimators=100, learning_rate=0.05, random_state=42, verbosity=-1))
    ])
    
    model_pipeline.fit(X_train, y_train)
    predictions = model_pipeline.predict(X_test)
    
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    
    print(f"R2 Score: {r2:.4f}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    
    joblib.dump(model_pipeline, MODEL_PATH)
    print("LightGBM pipeline saved successfully.")

if __name__ == "__main__":
    train_model()