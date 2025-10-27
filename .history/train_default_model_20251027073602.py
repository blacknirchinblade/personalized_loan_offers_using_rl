import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
from xgboost import XGBClassifier
import joblib
import os

print("--- Step 1 (v2): Default Model Training (with Categoricals) ---")

# --- 1. Setup File Names ---
DATA_FILE = 'accepted_2007_to_2018Q4.csv'
MODEL_FILE = 'default_model_v2.joblib'
PREPROCESSOR_FILE = 'preprocessor_v2.joblib'
FEATURES_FILE = 'model_features_v2.joblib'

# --- 2. Load and Prepare Data ---
print(f"Attempting to load data from: {DATA_FILE}")
try:
    df = pd.read_csv(DATA_FILE, low_memory=False)
except FileNotFoundError:
    print(f"Error: {DATA_FILE} not found. Please place it in the directory.")
    exit()

print("Filtering for completed loans...")
model_df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
model_df['is_default'] = (model_df['loan_status'] == 'Charged Off').astype(int)

# --- 3. Feature Engineering ---
print("Performing feature engineering...")
model_df['term'] = model_df['term'].str.strip().str.replace(' months', '').astype(float)
# Clean 'emp_length' (e.g., '< 1 year' -> 0, '10+ years' -> 10)
model_df['emp_length'] = model_df['emp_length'].str.replace(r'< 1 year', '0 years', regex=True)
model_df['emp_length'] = model_df['emp_length'].str.replace(r'10\+ years', '10 years', regex=True)
model_df['emp_length'] = model_df['emp_length'].str.replace(r' years?', '', regex=True)
# Handle 'n/a' by converting to NaN, then dropna will catch it
model_df['emp_length'] = pd.to_numeric(model_df['emp_length'], errors='coerce')

# --- Define our feature set ---
numeric_features = [
    'loan_amnt', 
    'int_rate', 
    'annual_inc', 
    'dti', 
    'fico_range_low', 
    'term',
    'emp_length'
]
categorical_features = [
    'home_ownership', 
    'verification_status', 
    'purpose'
]
target = 'is_default'
all_features = numeric_features + categorical_features

# Drop rows with missing values in our selected columns
model_df = model_df[all_features + [target]].dropna()
print(f"Final modeling data shape after cleaning: {model_df.shape}")

# Save the column order for our environment
joblib.dump(all_features, FEATURES_FILE)
print(f"Feature list saved to {FEATURES_FILE}")

# --- 4. Create Preprocessing Pipeline ---
# This pipeline will scale numbers and one-hot-encode categories
numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore', sparse_output=False)

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ],
    remainder='drop' 
)

# --- 5. Split Data ---
X = model_df[all_features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
X_train_sub, X_val, y_train_sub, y_val = train_test_split(
    X_train, y_train, test_size=0.1, random_state=42, stratify=y_train
)

# --- 6. Train XGBoost Default Model ---
print("\nTraining the XGBoost Default Model (with new features)...")
scale_pos_weight = y_train_sub.value_counts()[0] / y_train_sub.value_counts()[1]
print(f"Calculated scale_pos_weight: {scale_pos_weight:.2f}")

# Create a full pipeline: Preprocessing + Model
default_model_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        tree_method="hist",
        device="cuda",
        scale_pos_weight=scale_pos_weight,
        early_stopping_rounds=20
    ))
])

# --- Fit the *entire* pipeline ---
# We must manually preprocess the validation set for early stopping
print("Fitting preprocessor...")
preprocessor.fit(X_train_sub) # Fit preprocessor on (sub)training data
X_val_processed = preprocessor.transform(X_val) # Transform validation data

print("Fitting full model pipeline...")
default_model_pipeline.fit(
    X_train_sub, 
    y_train_sub, 
    classifier__eval_set=[(X_val_processed, y_val)],
    classifier__verbose=50
)
print("Model training complete.")

# --- 7. Evaluate and Save Model ---
y_pred = default_model_pipeline.predict(X_test)
print("\n--- Default Model Evaluation (v2) ---")
print(classification_report(y_test, y_pred, target_names=['Fully Paid (0)', 'Defaulted (1)']))

# Save the *entire* fitted pipeline
joblib.dump(default_model_pipeline, MODEL_FILE)
# Save the *fitted* preprocessor for the acceptance model
joblib.dump(preprocessor, PREPROCESSOR_FILE)

print(f"\nModel pipeline saved as '{MODEL_FILE}'")
print(f"Preprocessor saved as '{PREPROCESSOR_FILE}'")
print("--- Step 1 (v2) Complete ---")