import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib
import os

print("--- Step 2: Acceptance Model Training ---")

# --- 1. Load Data (reuse from Step 1) ---
DATA_FILE = 'accepted_2007_to_2018Q4.csv'
FEATURES_FILE = 'model_features.joblib'
ACCEPTANCE_MODEL_FILE = 'acceptance_model.joblib'

print(f"Attempting to load data from: {DATA_FILE}")
try:
    df = pd.read_csv(DATA_FILE, low_memory=False)
    features = joblib.load(FEATURES_FILE)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} or {FEATURES_FILE} not found.")
    print("Please run 'train_default_model.py' first.")
    exit()

print("Filtering for completed loans...")
model_df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
model_df['term'] = model_df['term'].str.strip().str.replace(' months', '').astype(float)
model_df = model_df[features].dropna() # Only need features this time

print(f"Loaded {model_df.shape[0]} accepted loans for synthetic data generation.")

# --- 2. Create Synthetic Acceptance Dataset ---
# Logic:
# 1. The 'int_rate' in the data is the rate they ACCEPTED. This is a "Positive" sample (1).
# 2. We'll create a synthetic "Negative" sample (0) by showing the *same customer*
#    a much higher, "unreasonable" rate.

print("Generating synthetic dataset...")
# Keep the original accepted loans
positive_samples = model_df.copy()
positive_samples['accepted'] = 1

# Create synthetic rejected loans
negative_samples = model_df.copy()
# Add a significant "rejection" markup (e.g., 8 percentage points higher)
negative_samples['int_rate'] = negative_samples['int_rate'] + 8.0 
negative_samples['accepted'] = 0

# Combine them
synth_df = pd.concat([positive_samples, negative_samples])
print(f"Synthetic dataset created. Shape: {synth_df.shape}")

# --- 3. Train Acceptance Model ---
# The features now include 'int_rate' (our action)
acceptance_features = features.copy() # ['loan_amnt', 'int_rate', 'annual_inc', ...]
target = 'accepted'

X = synth_df[acceptance_features].values
y = synth_df[target].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# A simple Logistic Regression is perfect for this probability model
# We use a Pipeline to scale the data first
acceptance_model = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(random_state=42, solver='saga')) # 'saga' is good for large data
])

print("Training the Acceptance Model...")
acceptance_model.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate and Save ---
y_pred = acceptance_model.predict(X_test)
print("\n--- Acceptance Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Accepted (1)']))

joblib.dump(acceptance_model, ACCEPTANCE_MODEL_FILE)
print(f"\nModel saved as '{ACCEPTANCE_MODEL_FILE}'")
print("--- Step 2 Complete ---")