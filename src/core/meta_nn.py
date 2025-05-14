import os
import joblib
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import StandardScaler


class MetaNeuralClassifier:
    def __init__(self):
        self.model = None
        self.scaler = None

    def build_model(self, input_dim):
        model = Sequential()
        model.add(Dense(64, activation="relu", input_dim=input_dim))
        model.add(Dropout(0.2))
        model.add(Dense(32, activation="relu"))
        model.add(Dropout(0.1))
        model.add(Dense(1, activation="sigmoid"))

        model.compile(
            optimizer=Adam(learning_rate=0.001),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )
        return model

    def fit(self, X, y, epochs=15, batch_size=64):
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        self.model = self.build_model(X_scaled.shape[1])
        early_stop = EarlyStopping(
            monitor="val_loss", patience=5, restore_best_weights=True
        )

        self.model.fit(
            X_scaled,
            y,
            validation_split=0.2,
            epochs=epochs,
            batch_size=batch_size,
            callbacks=[early_stop],
            verbose=1,
        )

    def predict(self, X):
        X_scaled = self.scaler.transform(X)
        preds = self.model.predict(X_scaled, verbose=0).flatten()
        return (preds > 0.5).astype(int)

    def predict_proba(self, X):
        X_scaled = self.scaler.transform(X)
        return self.model.predict(X_scaled, verbose=0).flatten()

    def save(self, path, version):
        if not os.path.exists(path):
            os.makedirs(path)
        self.model.save(os.path.join(path, f"meta_model_{version}.keras"))
        joblib.dump(self.scaler, os.path.join(path, f"meta_scaler_{version}.pkl"))

    def load(self, path, version):
        self.model = load_model(os.path.join(path, f"meta_model_{version}.keras"))
        self.scaler = joblib.load(os.path.join(path, f"meta_scaler_{version}.pkl"))
