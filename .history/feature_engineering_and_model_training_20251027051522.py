from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, accuracy_score
import joblib

# 1. 🧹 Select features (our "State")
# We'll start with a simple, common set of features
features = [
    'loan_amnt',      # The amount of the loan
    'int_rate',       # The interest rate (our future "Action")
    'annual_inc',     # The borrower's annual income
    'dti',            # Debt-to-Income ratio
    'fico_range_low', # FICO credit score
    'term'            # The term of the loan (e.g., 36 or 60 months)
]

# Our target variable
target = 'is_default'


df = pd.read_csv('accepted_2007_to_2018Q4.csv', low_memory=False, parse_dates=['issue_d'])
# Filter to only completed loans (fully paid or defaulted)
compleeted_loans_df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
# 2. Create the final modeling DataFrame
# Let's clean up the 'term' column first (e.g., " 36 months" -> 36)
completed_loans_df['term'] = completed_loans_df['term'].str.strip().str.replace(' months', '').astype(float)

# Select only our features and target
model_df = completed_loans_df[features + [target]].copy()

# 3. Handle Missing Values (simple way)
# For this first pass, we'll just drop any rows with missing data
original_rows = model_df.shape[0]
model_df = model_df.dropna()
print(f"\nDropped {original_rows - model_df.shape[0]} rows with missing values.")
print(f"Final modeling data shape: {model_df.shape}")

# 4. Split Data
X = model_df[features]
y = model_df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# 5. Train the "Default Model"
# We use GradientBoosting. It's powerful and good for imbalanced data.
# We'll train on a sample to speed things up.
# For a real project, you'd use the full set and tune hyperparameters.
print("\nTraining the Default Model (this may take a few minutes)...")
default_model = GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3, random_state=42, n_iter_no_change=5, validation_fraction=0.1)

# Let's train on a 200k sample to make it fast
X_train_sample = X_train.sample(n=min(200000, X_train.shape[0]), random_state=42)
y_train_sample = y_train[X_train_sample.index]

default_model.fit(X_train_sample, y_train_sample)
print("Model training complete.")

# 6. Evaluate Model
y_pred = default_model.predict(X_test)
print("\n--- Default Model Evaluation ---")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['Fully Paid (0)', 'Defaulted (1)']))

# 7. Save the Model
model_filename = 'default_model.joblib'
joblib.dump(default_model, model_filename)
print(f"\nModel saved as '{model_filename}'")