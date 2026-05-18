import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("Datasets/final_dataset.csv")

X = df.drop("Label", axis=1)
y = df["Label"]

# -------------------------------
# SPLIT DATA
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# SCALING (IMPORTANT)
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save scaler
joblib.dump(scaler, "Models/scaler.pkl")

# -------------------------------
# DEFINE MODELS
# -------------------------------
models = {
    "logistic_regression": LogisticRegression(),
    "svm": SVC(probability=True),
    "random_forest": RandomForestClassifier(n_estimators=100),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
best_model = None
best_accuracy = 0

# -------------------------------
# TRAIN & SAVE EACH MODEL
# -------------------------------
for name, model in models.items():
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"Accuracy: {acc:.4f}")

    # Save model
    model_filename = f"Models/{name}.pkl"
    joblib.dump(model, model_filename)
    print(f"Saved: {model_filename}")

    results.append([name, acc, prec, rec, f1])

    # Track best model
    if acc > best_accuracy:
        best_accuracy = acc
        best_model = model
        best_model_name = name

# -------------------------------
# VOTING CLASSIFIER
# -------------------------------
voting_model = VotingClassifier(
    estimators=[
        ("lr", models["logistic_regression"]),
        ("rf", models["random_forest"]),
        ("svm", models["svm"])
    ],
    voting='soft'
)

print("\nTraining: voting_classifier")

voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred)
rec = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {acc:.4f}")

# Save voting model
joblib.dump(voting_model, "Models/voting_classifier.pkl")
print("Saved: voting_classifier.pkl")

results.append(["voting_classifier", acc, prec, rec, f1])

# Check if voting is best
if acc > best_accuracy:
    best_model = voting_model
    best_model_name = "voting_classifier"

# -------------------------------
# SAVE BEST MODEL
# -------------------------------
joblib.dump(best_model, "Models/best_model.pkl")
print(f"\nBest Model: {best_model_name}")
print("Saved as: best_model.pkl")

# -------------------------------
# SAVE RESULTS
# -------------------------------
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

results_df.to_csv("Datasets/model_results.csv", index=False)

print("\nFinal Results:")
print(results_df)