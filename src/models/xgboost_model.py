import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
import os
from sklearn.model_selection import train_test_split

os.makedirs("trained_models", exist_ok=True)


def load_data():
    X = pd.read_csv("processed/X_train.csv")
    y = pd.read_csv("processed/y_train.csv").values.ravel()

    # 🔥 CRITICAL FIX 1: remove all object columns
    X = X.select_dtypes(include=[np.number])

    # 🔥 CRITICAL FIX 2: force clean numeric matrix
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X, y


def train():
    X, y = load_data()

    print("Final X shape:", X.shape)
    print("Dtypes:", X.dtypes.value_counts())

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    model = xgb.XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="auc",
        tree_method="hist"
    )

    model.fit(X_tr, y_tr)

    joblib.dump(model, "trained_models/xgboost.pkl")

    print("XGBoost trained successfully!")


if __name__ == "__main__":
    train()