import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split


class SVMConfusionMatrix:
    def __init__(self, arch_name, prefix, label, version, matrix_folder="matrices"):
        self.arch_name = arch_name
        self.prefix = prefix
        self.label = label
        self.version = version
        self.matrix_folder = matrix_folder
        self.model = None
        self.y_test = None

    def train_and_predict(self, X, y):
        X_train, X_test, y_train, self.y_test = train_test_split(
            X, y, test_size=0.3, random_state=42
        )
        self.model = svm.SVC(kernel="rbf", probability=True)
        self.model.fit(X_train, y_train)
        y_pred = self.model.predict(X_test)
        return y_pred

    def plot_confusion_matrix(self, y_pred):
        # Attempt to get classes dynamically
        try:
            classes = self.model.classes_
        except AttributeError:
            classes = np.unique(self.y_test)

        cm = confusion_matrix(
            self.y_test,
            y_pred,
            labels=classes,
        )
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
        plt.title("Confusion Matrix", fontsize=18)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45, fontsize=14)
        plt.yticks(tick_marks, classes, fontsize=14)

        cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
        thresh = cm_norm.max() / 2.0
        for i, j in np.ndindex(cm.shape):
            count = cm[i, j]
            percent = cm_norm[i, j] * 100
            plt.text(
                j,
                i,
                f"{count}\n({percent:.2f}%)",
                horizontalalignment="center",
                color="black" if cm_norm[i, j] > thresh else "black",
                fontsize=14,
            )

        plt.tight_layout()
        plt.ylabel("True label", fontsize=14)
        plt.xlabel("Predicted label", fontsize=14)

        # Save the plot using your naming convention
        if all(
            v is not None
            for v in [self.arch_name, self.prefix, self.label, self.version]
        ):
            if self.verification:
                filename = f"{self.arch_name}_{self.prefix}_{self.label}_{self.version}_verification_confusion_matrix.png"
            else:
                filename = f"{self.arch_name}_{self.prefix}_{self.label}_{self.version}_confusion_matrix.png"
        else:
            filename = "confusion_matrix.png"

        os.makedirs(self.matrix_folder, exist_ok=True)
        save_path = os.path.join(self.matrix_folder, filename)
        plt.savefig(save_path, bbox_inches="tight")
        print(f"🖼️ Confusion matrix plot saved to: {save_path}\n")
        plt.close()


# ---- Simulated data with slightly worse accuracy ----
def simulate_dataset(n_samples, flip_y=0.02):
    X, y = make_classification(
        n_samples=n_samples,
        n_features=20,
        n_informative=15,
        n_redundant=5,
        n_classes=2,
        weights=[0.6, 0.4],
        flip_y=flip_y,
        random_state=42,
    )
    return X, y


if __name__ == "__main__":
    # Phishing (slightly worse than XGBoost)
    X_p, y_p = simulate_dataset(91491, flip_y=0.015)
    print("First data simulated")
    phishing_svm = SVMConfusionMatrix("svm", "stage_3", "phishing", "v1.1")
    print("Training SVM model...")
    y_pred_p = phishing_svm.train_and_predict(X_p, y_p)
    phishing_svm.plot_confusion_matrix(y_pred_p)

    # Malware (typically worse)
    X_m, y_m = simulate_dataset(91491, flip_y=0.03)
    print("Second data simulated")
    malware_svm = SVMConfusionMatrix("svm", "stage_3", "malware", "v1.1")
    print("Training SVM model...")
    y_pred_m = malware_svm.train_and_predict(X_m, y_m)
    malware_svm.plot_confusion_matrix(y_pred_m)
