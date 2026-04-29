import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import os
import json 

os.makedirs("outputs", exist_ok=True)

def load():
    iso = joblib.load("trained_models/isolation_forest.pkl")
    xgb = joblib.load("trained_models/xgboost.pkl")
    ae = tf.keras.models.load_model("trained_models/autoencoder.keras")

    with open("processed/feature_schema.json") as f:
        cols = json.load(f)

    return iso, xgb, ae, cols


def predict():
    X = pd.read_csv("processed/X_test.csv")

    iso, xgb_model, ae, cols = load()

    X = X.reindex(columns=cols, fill_value=0)

    iso_score = -iso.decision_function(X)
    xgb_score = xgb_model.predict_proba(X)[:, 1]
    ae_score = np.mean(np.square(X - ae.predict(X)), axis=1)

    # normalize safely
    def norm(x):
        return (x - x.min()) / (x.max() - x.min() + 1e-8)

    iso_score = norm(iso_score)
    ae_score = norm(ae_score)

    final = 0.3 * iso_score + 0.5 * xgb_score + 0.2 * ae_score

    print("SAMPLE OUTPUT:")
    print(final[:10])

    output = pd.DataFrame({
    "iso_score": iso_score,
    "xgb_score": xgb_score,
    "ae_score": ae_score,
    "final_score": final
    })
    output.to_csv("outputs/scores.csv", index=False)

    pd.DataFrame({"fraud_score": final}).to_csv(
        "outputs/final_predictions.csv",
        index=False
    )

    print("Saved to outputs/final_predictions.csv")


if __name__ == "__main__":
    predict()