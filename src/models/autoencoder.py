import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import StandardScaler
import joblib
import os

os.makedirs("trained_models", exist_ok=True)


def load_data():
    X = pd.read_csv("processed/X_train.csv")

    # 🔥 CRITICAL FIX: keep ONLY numeric
    X = X.select_dtypes(include=[np.number])

    # clean infinities + NaNs
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(0)

    return X


def build_autoencoder(input_dim):
    inp = keras.Input(shape=(input_dim,))

    x = layers.Dense(128, activation="relu")(inp)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(32, activation="relu")(x)

    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dense(128, activation="relu")(x)

    out = layers.Dense(input_dim, activation="linear")(x)

    model = keras.Model(inp, out)
    model.compile(optimizer="adam", loss="mse")

    return model


def train():
    X = load_data()

    print("Raw shape:", X.shape)

    # 🔥 STEP 1: SCALING (VERY IMPORTANT)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # save scaler for inference
    joblib.dump(scaler, "trained_models/ae_scaler.pkl")

    # convert back to DataFrame (clean shape handling)
    X_scaled = pd.DataFrame(X_scaled)

    model = build_autoencoder(X_scaled.shape[1])

    history = model.fit(
        X_scaled,
        X_scaled,
        epochs=10,
        batch_size=256,
        validation_split=0.1,
        shuffle=True
    )

    model.save("trained_models/autoencoder.keras")

    print("Autoencoder trained successfully!")


if __name__ == "__main__":
    train()