import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import os

print("--- Step 1: Default Model Training ---")

# --- 1. Load and Prepare Data ---
DATA_FILE = 'accepted_2007_to_2018Q4.csv'
MODEL_FILE = 'default_model.joblib'
FEATURES_FILE = 'model_features.joblib'

print(f"Attempting to load data from: {DATA_FILE}")
try:
    df = pd.read_csv(DATA_FILE, low_memory=False, parse_dates=['issue_d'])
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found.")
    print("Please download the data from Kaggle (wordsforthewise/lending-club) and place it here.")
    exit()

# Filter for completed loans
print("Filtering for completed loans ('Fully Paid' or 'Charged Off')...")
model_df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

# Create the binary target variable
model_df['is_default'] = (model_df['loan_status'] == 'Charged Off').astype(int)

# --- 2. Feature Engineering ---
print("Performing feature engineering...")
# Simple 'term' cleaning
model_df['term'] = model_df['term'].str.strip().str.replace(' months', '').astype(float)

# Define our state (features)
# We use the same features for both models for consistency
features = [
    'loan_amnt',      # Amount requested
    'int_rate',       # The rate they *accepted* (important for acceptance model)
    'annual_inc',     # Annual income
    'dti',            # Debt-to-Income ratio
    'fico_range_low', # FICO score
    'term'            # Loan term
]
target = 'is_default'

# Drop rows with missing values in our selected columns
model_df = model_df[features + [target]].dropna()
print(f"Final modeling data shape after cleaning: {model_df.shape}")

# Save the column order for our environment later
joblib.dump(features, FEATURES_FILE)
print(f"Feature list saved to {FEATURES_FILE}")

# --- 3. Split Data ---
X = model_df[features]
y = model_df[target]

# We split the *full dataset* into train and test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# We split the *training set* again to get a validation set for early stopping
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

print(f"Full training set size: {X_train_sub.shape[0]}")
print(f"Validation set size: {X_val.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# --- 4. Train XGBoost Default Model ---
print("\nTraining the XGBoost Default Model (with imbalance fix)...")

# Calculate scale_pos_weight to handle imbalance
scale_pos_weight = y_train_sub.value_counts()[0] / y_train_sub.value_counts()[1]
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

default_model = XGBClassifier(
    n_estimators=500,     # More trees, since we have early stopping
    learning_rate=0.05,
    max_depth=5,
    random_state=42,
    tree_method="hist",   # Use 'hist' for CPU. Use 'gpu_hist' if you have NVIDIA GPU
    device="cuda",        # Set to 'cuda' for GPU, or 'cpu'
    scale_pos_weight=scale_pos_weight,
    early_stopping_rounds=20 # Stop after 20 rounds if no improvement
)

# Train using the full sub-training set and evaluate on the validation set
default_model.fit(
    X_train_sub, 
    y_train_sub, 
    eval_set=[(X_val, y_val)],
    verbose=50 # Print progress every 50 trees
)
print("Model training complete.")

# --- 5. Evaluate and Save Model ---
y_pred = default_model.predict(X_test)
print("\n--- Default Model Evaluation (XGBoost - Fixed) ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Fully Paid (0)', 'Defaulted (1)']))

joblib.dump(default_model, MODEL_FILE)
print(f"\nModel saved as '{MODEL_FILE}'")
print("--- Step 1 Complete ---")