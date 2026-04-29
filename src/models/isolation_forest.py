import pandas as pd
import joblib
from sklearn.ensemble import IsolationForest
import os
import numpy as np
import json

os.makedirs("trained_models", exist_ok=True)


def load_data():
    X = pd.read_csv("processed/X_train.csv")

    # force same schema order
    with open("processed/feature_schema.json") as f:
        cols = json.load(f)

    X = X[cols]   # 🔥 enforce exact match

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)
    return X

def train():
    X = load_data()

    print("Final training shape:", X.shape)
    print("Dtype check:", X.dtypes.value_counts())

    model = IsolationForest(
        n_estimators=150,
        contamination=0.02,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X)

    joblib.dump(model, "trained_models/isolation_forest.pkl")

    print("Isolation Forest trained successfully!")


if __name__ == "__main__":
    train()