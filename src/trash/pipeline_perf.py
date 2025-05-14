import os
import time
import pandas as pd
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_auc_score,
)
from core.validator import load_saved_split
from pipeline import DomainClassifier

# Configuration
STAGE = 1
VERIFICATION = True
LABELS = ["phishing", "malware"]
SAVE_PATH = "tex_sources/pipeline_verif.tex"

# Ensure output directory exists
os.makedirs("tex_sources", exist_ok=True)

# Initialize list for storing results
results = []

# Evaluate each label separately
for label in LABELS:
    print(f"\n=== Evaluating {label.upper()} (Stage {STAGE}) ===")

    # Load verification data for given label
    x_data, y_data = load_saved_split(STAGE, label, folder="./data/", verification=VERIFICATION)
    x_data = x_data[:10000]  # Limit data for performance
    y_data = y_data[:10000]

    # Initialize classifier
    clf = DomainClassifier(data_sample=x_data, label=label)
    clf.determine_stage(x_data)

    # Initialize metrics
    y_true = []
    y_pred = []
    y_proba = []

    # Measure inference time
    start_time = time.time()

    # Perform classification
    for x, true_label in zip(x_data, y_data):
        result = clf.classify_proba(x)
        final_proba = result["final_proba"]
        pred_label = 1 if final_proba > 0.5 else 0

        y_true.append(true_label)
        y_pred.append(pred_label)
        y_proba.append(final_proba)

    elapsed = time.time() - start_time
    speed = len(y_true) / elapsed

    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Compute evaluation metrics
    acc = (tp + tn) / (tp + tn + fp + fn)
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) > 0 else 0.0
    auc = roc_auc_score(y_true, y_proba)

    # Store results
    stage_results = {
        "Stage": STAGE,
        "Label": label,
        "Accuracy": acc,
        "Precision": prec,
        "Recall": recall,
        "F1 Score": f1,
        "ROC AUC": auc,
        "Domains/sec": round(speed, 2),
    }

    results.append(stage_results)
    print(f"Stage results: {stage_results}")

# Export results to LaTeX table
with open(SAVE_PATH, "w") as f:
    for result in results:
        f.write(f"""\\begin{{table}}[H]
                    \\centering
                    \\begin{{tabular}}{{|l|c|}}
                    \\hline
                    \\textbf{{Metrika}} & \\textbf{{{result['Label'].capitalize()}}} \\\\
                    \\hline
                    Přesnost (Accuracy) & \\texttt{{{result['Accuracy']:.4f}}} \\\\
                    Precision (Přesnost) & \\texttt{{{result['Precision']:.4f}}} \\\\
                    Recall (Úplnost) & \\texttt{{{result['Recall']:.4f}}} \\\\
                    F1 Skóre & \\texttt{{{result['F1 Score']:.4f}}} \\\\
                    ROC AUC & \\texttt{{{result['ROC AUC']:.4f}}} \\\\
                    \\hline
                    \\end{{tabular}}
                    \\caption{{Výsledky klasifikace {result['Label']} domén – verifikační sada}}
                    \\label{{tab:final_pipeline_ver_{result['Label']}_{result['Stage']}}}
                    \\end{{table}}
                    """)

print(f"Saved LaTeX results to: {SAVE_PATH}")
