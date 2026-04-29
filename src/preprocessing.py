import pandas as pd
import numpy as np
import os
import joblib
from sklearn.preprocessing import LabelEncoder
import json

DATA_PATH = "data/"
OUT_PATH = "processed/"

os.makedirs(OUT_PATH, exist_ok=True)


def load_data():
    train_tr = pd.read_csv(DATA_PATH + "train_transaction.csv")
    train_id = pd.read_csv(DATA_PATH + "train_identity.csv")

    test_tr = pd.read_csv(DATA_PATH + "test_transaction.csv")
    test_id = pd.read_csv(DATA_PATH + "test_identity.csv")

    train = train_tr.merge(train_id, on="TransactionID", how="left")
    test = test_tr.merge(test_id, on="TransactionID", how="left")

    return train, test


def reduce_memory(df):
    for col in df.columns:
        if df[col].dtype != "object":
            df[col] = pd.to_numeric(df[col], downcast="float")
    return df


def basic_cleaning(df):
    # numeric fill
    num_cols = df.select_dtypes(include=["number"]).columns
    df[num_cols] = df[num_cols].fillna(df[num_cols].median())

    # categorical fill (SAFE)
    cat_cols = df.select_dtypes(include=["object"]).columns
    for col in cat_cols:
        df[col] = df[col].astype(str).fillna("missing")

    return df


def feature_engineering(df):
    df = df.copy()

    if "TransactionAmt" in df.columns:
        df["TransactionAmt_log"] = np.log1p(df["TransactionAmt"])

    if "card1" in df.columns and "TransactionAmt" in df.columns:
        df["card1_amt_mean"] = df.groupby("card1")["TransactionAmt"].transform("mean")

    return df


def encode(train, test):
    encoders = {}

    cat_cols = list(set(train.select_dtypes(include=["object"]).columns) & set(test.columns))

    for col in cat_cols:
        le = LabelEncoder()

        train[col] = train[col].astype(str)
        test[col] = test[col].astype(str)

        le.fit(pd.concat([train[col], test[col]]))

        train[col] = le.transform(train[col])
        test[col] = le.transform(test[col])

        encoders[col] = le

    joblib.dump(encoders, OUT_PATH + "encoders.pkl")

    return train, test


def preprocess():
    print("Loading data...")
    train, test = load_data()

    print("Reducing memory...")
    train = reduce_memory(train)
    test = reduce_memory(test)

    print("Cleaning...")
    train = basic_cleaning(train)
    test = basic_cleaning(test)

    print("Feature engineering...")
    train = feature_engineering(train)
    test = feature_engineering(test)

    print("Encoding...")
    train, test = encode(train, test)

    # 🔥 STEP 1: UNIFY COLUMN NAMES (CRITICAL FIX)
    train.columns = train.columns.str.replace("-", "_")
    test.columns = test.columns.str.replace("-", "_")

    # split features/target
    X_train = train.drop(columns=["isFraud"])
    y_train = train["isFraud"]

    # 🔥 STEP 2: FINAL SAFETY — REMOVE ANY NON-NUMERIC LEFTOVER
    X_train = X_train.select_dtypes(include=["number"])
    test = test.select_dtypes(include=["number"])

    # fill NaNs safely
    X_train = X_train.replace([np.inf, -np.inf], np.nan).fillna(0)
    test = test.replace([np.inf, -np.inf], np.nan).fillna(0)

    # 🔥 STEP 3: SAVE FEATURE SCHEMA (CRITICAL FOR ENSEMBLE)
    feature_cols = X_train.columns.tolist()

    with open(OUT_PATH + "feature_schema.json", "w") as f:
        json.dump(feature_cols, f)

    # save datasets
    X_train.to_csv(OUT_PATH + "X_train.csv", index=False)
    y_train.to_csv(OUT_PATH + "y_train.csv", index=False)
    test.to_csv(OUT_PATH + "X_test.csv", index=False)

    print("Preprocessing DONE! Schema saved.")


if __name__ == "__main__":
    preprocess()