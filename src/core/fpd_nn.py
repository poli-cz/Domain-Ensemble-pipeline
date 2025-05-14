# fpd_nn.py

import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import joblib
import os


class FPDNeuralNetwork:
    def __init__(self):
        self.scaler = StandardScaler()
        self.model = None

    def fit(self, X, y):
        # Stratified split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )

        # Scale inputs
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)

        # Define model
        self.model = Sequential(
            [
                Dense(64, input_dim=X_train.shape[1], activation="relu"),
                Dropout(0.2),
                Dense(32, activation="relu"),
                Dropout(0.1),
                Dense(1, activation="sigmoid"),
            ]
        )

        self.model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        # Train model
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )
        self.model.fit(
            X_train,
            y_train,
            validation_data=(X_val, y_val),
            epochs=12,
            batch_size=64,
            callbacks=[early_stop],
            verbose=1,
        )

    def predict_fp(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)

        X_scaled = self.scaler.transform(X)

        probs = self.model.predict(X_scaled, verbose=0).flatten()
        return (probs > 0.5).astype(int)

    def predict_fp_proba(self, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

    def correct_predictions(self, preds, X):
        if X.ndim == 1:
            X = X.reshape(1, -1)
        is_fp = self.predict_fp(X)
        return np.where((preds == 1) & (is_fp == 1), 0, preds)

    def save(self, path, label, stage):
        os.makedirs(path, exist_ok=True)
        print(f"saving fp detektor as fpd_scaler_{label}_{stage}.joblib")

        self.model.save(os.path.join(path, f"fpd_model_{label}_{stage}.keras"))
        joblib.dump(
            self.scaler, os.path.join(path, f"fpd_scaler_{label}_{stage}.joblib")
        )

    def load(self, path, label, stage):
        self.model = load_model(os.path.join(path, f"fpd_model_{label}_{stage}.keras"))
        self.scaler = joblib.load(
            os.path.join(path, f"fpd_scaler_{label}_{stage}.joblib")
        )
