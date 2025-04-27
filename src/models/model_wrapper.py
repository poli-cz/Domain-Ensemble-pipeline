import os
import joblib
from tensorflow import keras
from xgboost import XGBModel
from lightgbm import LGBMModel, LGBMClassifier


class ModelWrapper:
    def __init__(self, model_dir="models"):
        self.model_dir = model_dir
        os.makedirs(self.model_dir, exist_ok=True)

    def _model_path(self, arch_name, label, version, ext="model"):
        return os.path.join(self.model_dir, f"{arch_name}_{label}_{version}.{ext}")

    def _normalize_prefix(self, label: str) -> str:
        """
        Detects aggregation stages in labels and normalizes them to 'stage_1/2/3_[type]'.
        E.g., lex__agg_phishing -> stage_1_phishing
        """
        if (
            "lex_+dns_+ip_+tls_+geo_+rdap__agg" in label
            or "lex_+dns_+ip_+tls_+geo_+rdap_agg" in label
        ):
            return "stage_3"
        if "lex_+dns_+ip_+geo__agg" in label or "lex_+dns_+ip_+geo_agg" in label:
            return "stage_2"
        if "lex__agg" in label or "lex_agg" in label:
            return "stage_1"
        return label

    def save(self, model, arch_name, label, prefix, version, overwrite=True):
        prefix = self._normalize_prefix(prefix)
        print("Saving as stage:", prefix)
        combined_label = f"{prefix}_{label}" if prefix else label

        if isinstance(model, keras.Model):
            ext = "keras"
        elif isinstance(model, XGBModel):
            ext = "xgb"
        elif isinstance(model, LGBMModel):
            ext = "pkl"  # Now saving LGBM via joblib
        elif hasattr(model, "predict") and hasattr(model, "fit"):
            ext = "pkl"  # Generic sklearn-like
        else:
            raise ValueError("Unsupported model type")

        path = self._model_path(arch_name, combined_label, version, ext)

        if os.path.exists(path) and not overwrite:
            raise FileExistsError(f"Model already exists: {path}")

        if isinstance(model, keras.Model):
            model.save(path)
        elif isinstance(model, XGBModel):
            model.save_model(path)
        else:
            joblib.dump(model, path)  # LGBM and sklearn models

    def load(self, arch_name, label, prefix, version):
        prefix = self._normalize_prefix(prefix)
        combined_label = f"{prefix}_{label}" if prefix else label

        for ext in ["keras", "xgb", "pkl"]:
            path = self._model_path(arch_name, combined_label, version, ext)
            if os.path.exists(path):
                print(f"📦 Loading model from {path}")
                if ext == "keras":
                    return keras.models.load_model(path)
                elif ext == "xgb":
                    from xgboost import XGBClassifier

                    model = XGBClassifier()
                    model.load_model(path)
                    return model
                elif ext == "pkl":
                    return joblib.load(path)

        raise FileNotFoundError(
            f"Model not found for {arch_name}_{combined_label}_{version}"
        )

    def exists(self, arch_name, label, version):
        return any(
            os.path.exists(self._model_path(arch_name, label, version, ext))
            for ext in ["keras", "xgb", "pkl"]
        )

    def delete(self, arch_name, label, version):
        for ext in ["keras", "xgb", "pkl"]:
            path = self._model_path(arch_name, label, version, ext)
            if os.path.exists(path):
                os.remove(path)
                print(f"🗑️ Deleted model: {path}")
                return
        print(f"⚠️ Model not found: {arch_name}_{label}_{version}")

    def list_models(self):
        return [
            f
            for f in os.listdir(self.model_dir)
            if f.endswith((".keras", ".xgb", ".pkl"))
        ]
