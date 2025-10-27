# --- 1. Setup Kaggle API (Run this in Google Colab) ---
# Install the Kaggle library
!pip install kaggle

# Create a directory for the Kaggle API key
!mkdir -p ~/.kaggle

# ---
# IMPORTANT: Manually upload your 'kaggle.json' file to the '/root/.kaggle/' directory
# You can do this by clicking the "Files" icon on the left sidebar in Colab,
# navigating to /root/.kaggle/, and uploading your file.
# ---

# Set the permissions for the key
!chmod 600 ~/.kaggle/kaggle.json

print("Kaggle API setup complete.")

# --- 2. Download and Unzip the Data ---
# Download the dataset (this will take a minute or two)
!kaggle datasets download -d wordsforthewise/lending-club

# Unzip the main file
!unzip lending-club.zip accepted_2007_to_2018Q4.csv

print("Dataset downloaded and unzipped.")

# --- 3. Load and Prepare the Data ---
import pandas as pd
import numpy as np

# This is a large file, so 'low_memory=False' helps
# We'll also parse the 'issue_d' (issue date) as a datetime object
try:
    df = pd.read_csv('accepted_2007_to_2018Q4.csv', low_memory=False, parse_dates=['issue_d'])
    print(f"Data loaded successfully. Shape: {df.shape}")
except FileNotFoundError:
    print("File not found. Please ensure 'accepted_2007_to_2018Q4.csv' is in the correct directory.")

# --- 4. 🎯 Create the Target Variable ---
# This is the most important step for our helper model.
# The 'loan_status' column has many values (e.g., "Current", "In Grace Period").
# We only care about loans that are *finished*.
print("\nOriginal 'loan_status' values:")
print(df['loan_status'].value_counts())

# Filter for completed loans
completed_loans_df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()

# Create the binary target variable
# 'Charged Off' is a default (1), 'Fully Paid' is not (0).
completed_loans_df['is_default'] = (completed_loans_df['loan_status'] == 'Charged Off').astype(int)

print(f"\nFiltered data for completed loans. New shape: {completed_loans_df.shape}")
print("\nNew 'is_default' target variable counts:")
print(completed_loans_df['is_default'].value_counts())