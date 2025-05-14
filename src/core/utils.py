import pandas as pd
import numpy as np
import joblib


def safe_scale(X, arch_name=None, label=None, stage=None):
    def svm_convert(df):
        # convert to dataframe from numpy array
        df = pd.DataFrame(df)
        df = df.fillna(0)
        df = df.replace({True: 1, False: 0})
        return np.array(df)

    ### Variable preprocessing for different architectures and models ###
    if arch_name == "feedforward":
        X = joblib.load(f"scalers/{label}_{arch_name}_{stage}_scaler.joblib").transform(
            X
        )

    elif arch_name == "svm":
        print(
            f"🔄 Detected SVM, using: scalers/{label}_{arch_name}_{stage}_scaler.joblib scaler"
        )
        X = joblib.load(f"scalers/{label}_{arch_name}_{stage}_scaler.joblib").transform(
            X
        )

        X = svm_convert(X)

    elif arch_name == "attention_tls":
        # Load selected feature indices (e.g., [49, 50, ..., 72])

        # Apply scaler
        scaler_path = f"scalers/{label}_{arch_name}_{stage}_scaler.joblib"
        X = joblib.load(scaler_path).transform(X)

        # print shape of the data
        print(f"X_test shape: {X.shape}")
        print("Scaled and selected attention_tls input features")

    elif arch_name == "cnn":
        # Načti scaler
        scaler_path = f"scalers/{label}_{arch_name}_{stage}_scaler.joblib"
        X = joblib.load(scaler_path).transform(X)

        # Výpočet velikosti čtverce
        original_feature_size = X.shape[1]
        side_size = int(np.ceil(np.sqrt(original_feature_size)))
        padded_size = side_size**2
        padding = padded_size - original_feature_size

        # Padding nulami napravo
        if padding > 0:
            X = np.pad(
                X,
                ((0, 0), (0, padding)),
                mode="constant",
                constant_values=0,
            )

        # Reshape na (N, side, side, 1)
        X = X.reshape(-1, side_size, side_size, 1)
        print(
            f"[CNN] X_test reshaped to {X.shape} (side {side_size}, padded by {padding})"
        )

    return X


def safe_predict(model, X, arch_name=None, label=None, stage=None):
    """
    Evaluates and prints the model's performance metrics, including F1 score,
    confusion matrix, and a classification report. Also saves a confusion matrix plot.
    """

    X = safe_scale(X, arch_name=arch_name, label=label, stage=stage)

    y_pred = model.predict(X)

    # --- Safe prediction processing ---
    # Handle probability outputs (e.g., from neural networks)
    if np.issubdtype(y_pred.dtype, np.floating):
        if y_pred.ndim == 2 and y_pred.shape[1] > 1:
            # Multiclass or binary with probabilities -> argmax
            y_pred = np.argmax(y_pred, axis=1)
        else:
            # Single probability output -> threshold at 0.5
            y_pred = (y_pred >= 0.5).astype(float)

    return y_pred
