import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import joblib
import os

print("--- Step 2 (v2): Acceptance Model Training ---")

# --- 1. Load Data & Fitted Preprocessor ---
DATA_FILE = 'accepted_2007_to_2018Q4.csv'
ACCEPTANCE_MODEL_FILE = 'acceptance_model_v2.joblib'
PREPROCESSOR_FILE = 'preprocessor_v2.joblib'
FEATURES_FILE = 'model_features_v2.joblib'

try:
    preprocessor = joblib.load(PREPROCESSOR_FILE)
    all_features = joblib.load(FEATURES_FILE)
except FileNotFoundError:
    print(f"Error: Files not found. Please run 'train_default_model.py' (v2) first.")
    exit()

print("Loading and preparing data...")
df = pd.read_csv(DATA_FILE, low_memory=False)
model_df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

# Feature engineering (must be identical to default model script)
model_df['term'] = model_df['term'].str.strip().str.replace(' months', '').astype(float)
model_df['emp_length'] = model_df['emp_length'].str.replace(r'< 1 year', '0 years', regex=True)
model_df['emp_length'] = model_df['emp_length'].str.replace(r'10\+ years', '10 years', regex=True)
model_df['emp_length'] = model_df['emp_length'].str.replace(r' years?', '', regex=True)
model_df['emp_length'] = pd.to_numeric(model_df['emp_length'], errors='coerce')

model_df = model_df[all_features].dropna()
print(f"Loaded {model_df.shape[0]} accepted loans for synthetic data generation.")

# --- 2. Create Synthetic Acceptance Dataset ---
print("Generating synthetic dataset...")
positive_samples = model_df.copy()
positive_samples['accepted'] = 1

negative_samples = model_df.copy()
# Add a significant "rejection" markup
negative_samples['int_rate'] = negative_samples['int_rate'] + 8.0 
negative_samples['accepted'] = 0

synth_df = pd.concat([positive_samples, negative_samples])

# --- 3. Train Acceptance Model ---
target = 'accepted'
X = synth_df[all_features]
y = synth_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Create a full pipeline: Preprocessing + Model ---
acceptance_model_pipeline = Pipeline([
    # Use the *exact same* (already fitted) preprocessor
    ('preprocessor', preprocessor), 
    ('model', LogisticRegression(random_state=42, solver='saga', max_iter=200)) # Increased iter
])

print("Training the Acceptance Model...")
# Note: We are 'fitting' the pipeline, but the preprocessor step is already fitted
# and will only 'transform' the data.
acceptance_model_pipeline.fit(X_train, y_train)
print("Model training complete.")

# --- 4. Evaluate and Save ---
y_pred = acceptance_model_pipeline.predict(X_test)
print("\n--- Acceptance Model Evaluation (v2) ---")
print(classification_report(y_test, y_pred, target_names=['Rejected (0)', 'Accepted (1)']))

joblib.dump(acceptance_model_pipeline, ACCEPTANCE_MODEL_FILE)
print(f"\nModel pipeline saved as '{ACCEPTANCE_MODEL_FILE}'")
print("--- Step 2 (v2) Complete ---")