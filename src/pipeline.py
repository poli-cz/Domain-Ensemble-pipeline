import os
import io
import warnings
import sys
import joblib
import contextlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap


# ========== System Imports ==========
from models.model_wrapper import ModelWrapper
from core.meta_nn import MetaNeuralClassifier
from core.fpd_nn import FPDNeuralNetwork
from core.utils import safe_predict


# ========== Suppress Warnings and Logs ==========
@contextlib.contextmanager
def suppress_stdout():
    with contextlib.redirect_stdout(io.StringIO()):
        yield


# Suppress TensorFlow logs
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

# Suppress sklearn feature name warnings
warnings.filterwarnings("ignore", message="X does not have valid feature names")


ARCHITECTURES = ["cnn", "svm", "XgBoost", "Lgbm", "feedforward"]
VERSION = "v1.1"
FPD_MODEL_PATH_TEMPLATE = "./models/fpd_saved_model"
META_MODEL_PATH_TEMPLATE = "./models/meta_nn_model"


class DomainClassifier:
    def __init__(self, data_sample, label: str = "phishing"):
        self.stage = self.determine_stage(data_sample)
        print("Stage:", self.stage)
        self.label = label
        self.model_wrapper = ModelWrapper(model_dir="models")
        self.base_models = self._load_base_models()
        self.meta_model = self._load_meta_model()
        self.fpd_model = self._load_fpd_model()

    def _load_base_models(self):
        models = {}
        for arch in ARCHITECTURES:
            model = self.model_wrapper.load(
                arch_name=arch,
                label=self.label,
                prefix=f"stage_{self.stage}",
                version=VERSION,
            )
            models[arch] = model
        return models

    def determine_stage(self, feature_vector: np.ndarray):
        """
        Určuje fázi klasifikace na základě počtu dostupných příznaků ve vstupním vektoru.
        """
        n_features = (
            feature_vector.shape[0]
            if feature_vector.ndim == 1
            else feature_vector.shape[1]
        )

        if n_features < 62:
            raise ValueError(
                "Feature vector has too few columns for any phase (minimum is 62)."
            )
        elif n_features < 128:
            self.stage = 1
        elif n_features < 176:
            self.stage = 2
        elif n_features == 176:
            self.stage = 3
        else:
            print()

        print("Stage determined:", self.stage)
        return self.stage

    def _load_meta_model(self):
        meta_model = MetaNeuralClassifier()
        meta_model.load(META_MODEL_PATH_TEMPLATE, "v1.1")
        return meta_model

    def _load_fpd_model(self):
        fpd_model = FPDNeuralNetwork()
        fpd_model.load(FPD_MODEL_PATH_TEMPLATE, self.label, self.stage)
        return fpd_model

    def _predict_arch(self, model, x, architecture):
        # failsafe
        if x.ndim == 1:
            x = x.reshape(1, -1)

        y_pred = safe_predict(model, x, architecture, self.label, self.stage)

        return y_pred

    def classify(self, feature_vector: np.ndarray):
        """Returns binary predictions: meta + FPD corrected"""
        all_preds = []
        for arch, model in self.base_models.items():
            preds = self._predict_arch(model, feature_vector, arch)
            all_preds.append(preds)

        meta_input = np.vstack(all_preds).T
        meta_preds = self.meta_model.predict(meta_input)

        corrected_preds = self.fpd_model.correct_predictions(meta_preds, feature_vector)

        return {
            "stage": self.stage,
            "meta_pred": meta_preds.tolist(),
            "corrected_pred": corrected_preds.tolist(),
        }

    def classify_proba(self, feature_vector: np.ndarray):
        """Returns float probabilities"""

        feature_vector = self.chop_feature_vector(feature_vector)

        all_preds = []
        for arch, model in self.base_models.items():
            preds = self._predict_arch(model, feature_vector, arch)
            all_preds.append(preds)

        meta_input = np.vstack(all_preds).T

        # add 10 features from feture vector to meta input
        meta_input_final = np.hstack((meta_input, feature_vector[:10].reshape(1, -1)))

        with suppress_stdout():
            meta_proba = float(self.meta_model.predict_proba(meta_input_final)[0])

        if meta_proba > 0.5:
            with suppress_stdout():
                fpd_proba = float(self.fpd_model.predict_fp_proba(feature_vector)[0])
        else:
            fpd_proba = 0.0

        final_proba = meta_proba
        # if result is positive and fpd is over 0,5 than it if false positive
        if meta_proba > 0.5 and fpd_proba > 0.5:
            # we have false positive
            final_proba = 0.0

        return {
            "partial_preds": [float(p) for p in all_preds],
            "stage": self.stage,
            "meta_proba": round(meta_proba, 4),
            "fpd_proba": round(fpd_proba, 4),
            "final_proba": round(final_proba, 4),
        }

    def chop_feature_vector(self, feature_vector: np.ndarray):
        """
        Chops the feature vector to the maximum size of 128 features.
        """
        if self.stage == 1:
            # if len is over 62, chop to 62
            if feature_vector.shape[0] > 62:
                feature_vector = feature_vector[:62]
            elif feature_vector.shape[0] < 62:
                # if len is under 62, pad with zeros
                feature_vector = np.pad(
                    feature_vector, (0, 62 - feature_vector.shape[0]), "constant"
                )

        elif self.stage == 2:
            if feature_vector.shape[0] > 128:
                feature_vector = feature_vector[:128]

        elif self.stage == 3:
            if feature_vector.shape[0] > 176:
                feature_vector = feature_vector[:176]
        else:
            raise ValueError("Invalid stage. Stage must be 1, 2, or 3.")

        return feature_vector

    def classify_batch(self, X: np.ndarray):
        """Run batch classification on input array and return results per model."""
        results = {}
        for arch, model in self.base_models.items():
            preds = []
            for x in X:
                pred = self._predict_arch(model, x, arch)
                preds.append(pred[0])
            results[arch] = preds
        return results

    def explain_shap(
        self, X: np.ndarray, columns: list[str], output_dir="shap_outputs"
    ):
        """Compute SHAP explanations for each base model and save them as PNG and TXT."""
        os.makedirs(output_dir, exist_ok=True)
        for arch, model in self.base_models.items():
            print(f"Generating SHAP for {arch}...")
            self._explain_model_shap(X, columns, arch, model, output_dir)

    def _explain_model_shap(
        self, X: np.ndarray, columns: list[str], arch: str, model, output_dir: str
    ):
        # Handle feedforward model scaling
        if arch == "feedforward" or arch == "svm":
            scaler_path = f"scalers/{self.label}_{arch}_{self.stage}_scaler.joblib"
            scaler = joblib.load(scaler_path)
            X_scaled = scaler.transform(X)
        else:
            X_scaled = X

        try:
            # For Keras or sklearn-based models
            explainer = shap.Explainer(model.predict, X_scaled)
            shap_values = explainer(X_scaled)

            # Save image
            plt.figure()
            shap.summary_plot(
                shap_values, features=X_scaled, feature_names=columns, show=False
            )
            plt.savefig(os.path.join(output_dir, f"shap_{arch}.png"))
            plt.close()

            # Save textual summary
            with open(os.path.join(output_dir, f"shap_{arch}.txt"), "w") as f:
                shap.summary_plot(
                    shap_values, features=X_scaled, feature_names=columns, show=False
                )
                vals = np.abs(shap_values.values).mean(0)
                for i in np.argsort(vals)[::-1]:
                    f.write(f"{columns[i]}: {vals[i]:.4f}\n")

        except Exception as e:
            print(f"[WARN] SHAP failed for {arch}: {e}")
