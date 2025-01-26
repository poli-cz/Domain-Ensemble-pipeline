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

# 1. Create a mock dataset
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

# 2. Simple classifiers
classifiers = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(max_depth=3),
    "SVM": SVC(kernel="linear", probability=True),
}

# Train the base classifiers
predictions = {}
accuracies = {}
for name, clf in classifiers.items():
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)
    predictions[name] = pred
    accuracies[name] = accuracy_score(y_test, pred)

# 3. Combine methods
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


# Function to plot decision boundaries
def plot_decision_boundaries(X, y, model, ax, title):
    h = 0.02  # Step size in the mesh
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    ax.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    scatter = ax.scatter(X[:, 0], X[:, 1], c=y, edgecolors="k", cmap="coolwarm", s=30)
    ax.set_title(title)
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    return scatter


# 4. Visualization of the results and decision boundaries

# Plot for plain methods (simple classifiers)
plt.figure(figsize=(12, 4))

for i, (name, clf) in enumerate(classifiers.items(), start=1):
    ax = plt.subplot(1, 3, i)
    plot_decision_boundaries(
        X_test, y_test, clf, ax, f"{name} (Acc: {accuracies[name]:.3f})"
    )

plt.tight_layout()
plt.show()

# Plot for combined methods
plt.figure(figsize=(12, 4))

methods = {
    "Bagging (Random Forest)": (bagging, bagging_accuracy),
    "Boosting (AdaBoost)": (boosting, boosting_accuracy),
    "Stacking": (stacking, stacking_accuracy),
}

for i, (name, (clf, acc)) in enumerate(methods.items(), start=1):
    ax = plt.subplot(1, 3, i)
    plot_decision_boundaries(X_test, y_test, clf, ax, f"{name} (Acc: {acc:.3f})")

plt.tight_layout()
plt.show()

# Print the accuracy of each model
for i, (name, pred) in enumerate(predictions.items(), start=1):
    print(f"{name} accuracy: {accuracies[name]:.3f}")

for i, (name, (pred, acc)) in enumerate(methods.items(), start=1):
    print(f"{name} accuracy: {acc:.3f}")
