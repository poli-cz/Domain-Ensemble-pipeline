import numpy as np
import matplotlib.pyplot as plt
import os


def plot_binary_confusion_matrix(tn, fp, fn, tp, name):
    cm = np.array([[tn, fp], [fn, tp]])
    classes = [0, 1]  # Binary classification: 0 = Negative class, 1 = Positive class

    plt.figure(figsize=(6, 6))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Reds)
    plt.title("Matice záměn", fontsize=18)
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, classes, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    thresh = cm.max() / 2.0

    for i, j in np.ndindex(cm.shape):
        count = cm[i, j]
        percent = cm_norm[i, j] * 100
        plt.text(
            j,
            i,
            f"{count}\n({percent:.2f}%)",
            horizontalalignment="center",
            color="black" if cm[i, j] > thresh else "black",
            fontsize=15,
        )

    plt.tight_layout()
    plt.ylabel("Skutečná třída", fontsize=14)
    plt.xlabel("Predikovaná třída", fontsize=14)

    filename = f"{name}.png"

    plt.savefig(filename, bbox_inches="tight")
    print(f"🖼️ Binary confusion matrix plot saved to: {filename}\n")
    plt.close()


plot_binary_confusion_matrix(
    tn=81489,  # True negatives
    fp=182,  # False positives
    fn=927,  # False negatives
    tp=15477,  # True positives
    name="cnn_stage_3_phishing_v1.1_confusion_matrix",
)

plot_binary_confusion_matrix(
    tn=81456,  # True negatives
    fp=187,  # False positives
    fn=3980,  # False negatives
    tp=6091,  # True positives
    name="cnn_stage_3_malware_v1.1_confusion_matrix",
)
