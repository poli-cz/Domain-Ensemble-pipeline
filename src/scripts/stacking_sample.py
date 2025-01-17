import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.ensemble import (
    RandomForestClassifier,
    AdaBoostClassifier,
    StackingClassifier,
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# 1. Vytvoření mock datové sady
X, y = make_classification(
    n_samples=1000,
    n_features=2,
    n_classes=2,
    n_redundant=0,
    n_clusters_per_class=2,
    flip_y=0.1,
    class_sep=0.8,
    random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 2. Jednoduché klasifikátory
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "SVM": SVC(kernel="linear", probability=True),
}

# Trénování základních klasifikátorů
predictions = {}
accuracies = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    predictions[name] = pred
    accuracies[name] = accuracy_score(y_test, pred)

# 3. Kombinace metodami
# Bagging - Random Forest
bagging = RandomForestClassifier(n_estimators=50, random_state=42)
bagging.fit(X_train, y_train)
bagging_pred = bagging.predict(X_test)
bagging_accuracy = accuracy_score(y_test, bagging_pred)

# Boosting - AdaBoost
boosting = AdaBoostClassifier(
    base_estimator=DecisionTreeClassifier(max_depth=1), n_estimators=50, random_state=42
)
boosting.fit(X_train, y_train)
boosting_pred = boosting.predict(X_test)
boosting_accuracy = accuracy_score(y_test, boosting_pred)

# Stacking
stacking = StackingClassifier(
    estimators=[
        ("lr", LogisticRegression()),
        ("dt", DecisionTreeClassifier(max_depth=3)),
        ("svc", SVC(kernel="linear", probability=True)),
    ],
    final_estimator=LogisticRegression(),
)
stacking.fit(X_train, y_train)
stacking_pred = stacking.predict(X_test)
stacking_accuracy = accuracy_score(y_test, stacking_pred)

# 4. Vizualizace výsledků
plt.figure(figsize=(12, 4))

# Referenční metody
for i, (name, pred) in enumerate(predictions.items(), start=1):
    plt.subplot(1, 3, i)
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=pred, cmap="coolwarm", s=30, edgecolor="k"
    )
    plt.title(f"{name} (Acc: {accuracies[name]:.3f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

# Kombinované metody
methods = {
    "Bagging (Random Forest)": (bagging_pred, bagging_accuracy),
    "Boosting (AdaBoost)": (boosting_pred, boosting_accuracy),
    "Stacking": (stacking_pred, stacking_accuracy),
}

plt.tight_layout()
plt.show()

# Přehled výsledků kombinovaných metod
plt.figure(figsize=(12, 4))

for i, (name, (pred, acc)) in enumerate(methods.items(), start=1):
    plt.subplot(1, 3, i)
    plt.scatter(
        X_test[:, 0], X_test[:, 1], c=pred, cmap="coolwarm", s=30, edgecolor="k"
    )
    plt.title(f"{name} (Acc: {acc:.3f})")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

plt.tight_layout()
plt.show()

for i, (name, pred) in enumerate(predictions.items(), start=1):
    print(f"{name} accuracy: {accuracies[name]:.3f}")

for i, (name, (pred, acc)) in enumerate(methods.items(), start=1):
    print(f"{name} accuracy: {acc:.3f}")
