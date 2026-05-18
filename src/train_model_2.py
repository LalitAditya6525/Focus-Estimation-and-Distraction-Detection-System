import os
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ML Models
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.neighbors import KNeighborsClassifier
from xgboost import XGBClassifier

# -------------------------------
# CREATE FOLDERS IF NOT EXIST
# -------------------------------
os.makedirs("Models", exist_ok=True)
os.makedirs("Datasets", exist_ok=True)

# -------------------------------
# LOAD DATA
# -------------------------------
df = pd.read_csv("Datasets/final_dataset.csv")

# -------------------------------
# HANDLE CATEGORICAL FEATURES
# -------------------------------
categorical_cols = ["GazeDirection", "Posture", "Emotion"]

encoders = {}
for col in categorical_cols:
    if col in df.columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        encoders[col] = le

# Save encoders
joblib.dump(encoders, "Models/Test2/encoders.pkl")

# -------------------------------
# FEATURES & TARGET
# -------------------------------
X = df.drop("Label", axis=1)
y = df["Label"]

# Encode target if needed
if y.dtype == 'object':
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    joblib.dump(label_encoder, "Models/Test2/label_encoder.pkl")

# -------------------------------
# SPLIT DATA
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# -------------------------------
# SCALING
# -------------------------------
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

joblib.dump(scaler, "Models/Test2/scaler.pkl")

# -------------------------------
# DEFINE MODELS
# -------------------------------
models = {
    "logistic_regression": LogisticRegression(max_iter=500),
    "svm": SVC(probability=True),
    "random_forest": RandomForestClassifier(n_estimators=150),
    "knn": KNeighborsClassifier(n_neighbors=5),
    "xgboost": XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
best_model = None
best_accuracy = 0

# -------------------------------
# TRAIN MODELS
# -------------------------------
for name, model in models.items():
    print(f"\nTraining: {name}")

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)

    print(f"Accuracy: {acc:.4f}")

    # Save model
    model_path = f"Models/Test2/{name}.pkl"
    joblib.dump(model, model_path)

    results.append([name, acc, prec, rec, f1])

    # Track best
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
    voting="soft"
)

print("\nTraining: voting_classifier")

voting_model.fit(X_train, y_train)
y_pred = voting_model.predict(X_test)

acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec = recall_score(y_test, y_pred, zero_division=0)
f1 = f1_score(y_test, y_pred, zero_division=0)

print(f"Accuracy: {acc:.4f}")

joblib.dump(voting_model, "Models/Test2/voting_classifier.pkl")

results.append(["voting_classifier", acc, prec, rec, f1])

if acc > best_accuracy:
    best_model = voting_model
    best_model_name = "voting_classifier"

# -------------------------------
# SAVE BEST MODEL
# -------------------------------
joblib.dump(best_model, "Models/Test2/best_model.pkl")

print(f"\n🏆 Best Model: {best_model_name}")

# -------------------------------
# SAVE RESULTS
# -------------------------------
results_df = pd.DataFrame(results, columns=[
    "Model", "Accuracy", "Precision", "Recall", "F1 Score"
])

results_df.to_csv("Models/Test2/model_results.csv", index=False)

print("\n📊 Final Results:")
print(results_df.sort_values(by="Accuracy", ascending=False))