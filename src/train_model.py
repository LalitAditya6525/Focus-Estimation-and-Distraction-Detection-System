import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import joblib

# Load dataset
df = pd.read_csv("Datasets/attention_detection_dataset_v1.csv")

# Preview
print(df.head())

# Encode categorical columns if any
for col in df.columns:
    if df[col].dtype == 'object' or df[col].dtype == 'str':
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

# Convert all columns to numeric 
df = df.astype('float64')

# Assume last column is target (Focus / Distracted)
X = df.iloc[:, :-1]
y = df.iloc[:, -1]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save model
joblib.dump(model, "Models/focus_model.pkl")

print("Model trained and saved successfully!")