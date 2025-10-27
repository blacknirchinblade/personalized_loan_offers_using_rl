# Personalized Loan Offers Using Reinforcement Learning

This repository contains code, notebooks and scripts used to prototype a system for predicting loan default risk and using reinforcement learning to personalize loan offers. It includes data download helpers, feature engineering, baseline models (XGBoost), and a prototype RL environment and agent.

## Highlights

- Processed Lending Club dataset (millions of records) for borrower risk modeling
- Baseline risk models trained with XGBoost (CPU/GPU)
- Prototype OpenAI Gym-like environment (`loan_env.py`) for RL agents to optimize offer decisions
- Cross-platform data download and permission-handling (Kaggle integration)

## Architecture (high level)

1. Data ingestion
   - `dataset_download.ipynb`: Notebook with cross-platform steps to download data from Kaggle and extract files into `data/`.
2. Feature engineering & baseline modeling
   - `feature_engineering_and_model_training.py`: Prepares features and trains an XGBoost default-risk model. Saves model artifacts as `.joblib`.
   - `train_default_model.py` and `train_acceptance_model.py`: Training scripts for baseline models.
3. RL environment & agent
   - `loan_env.py`: Gym-compatible environment that simulates loan offers and borrower responses.
   - `train_agent.py` / `test_agent.py`: Scripts to train/evaluate RL agents interacting with the environment.
4. Experiment/logs
   - TensorBoard logs live under `dqn_loan_tensorboard/` and `dqn_loan_tensorboard_v2/` (these are ignored in the repo).

## File Structure

Important files and folders:

```
dataset_download.ipynb       # Notebook: download and prepare dataset (Kaggle)
feature_engineering_and_model_training.py  # Feature prep and XGBoost training
train_default_model.py       # Script to train default prediction model
train_acceptance_model.py    # Script to train the acceptance model
loan_env.py                  # RL environment (Gym-like)
train_agent.py               # Train RL agent
test_agent.py                # Run/evaluate agent
data/                        # (ignored) where CSVs and extracted files should go
.gitignore
README.md
LICENSE
```

## Setup

Recommended: create a Python virtual environment (conda or venv). Example (conda):

```powershell
conda create -n loan-rl python=3.8 -y
conda activate loan-rl
pip install -r requirements.txt
```

If you don't have a `requirements.txt`, install the main packages:

```powershell
pip install pandas scikit-learn xgboost joblib kaggle kagglehub gym
```

Note: `xgboost` is optional but recommended for performance. For GPU acceleration install CUDA (see GPU notes below).

## Data (Kaggle)

- Get a Kaggle API token (`kaggle.json`) from your Kaggle account and place it in `%USERPROFILE%\.kaggle\kaggle.json` (Windows) or `~/.kaggle/kaggle.json` (Linux/Mac). The repository contains a notebook (`dataset_download.ipynb`) that shows cross-platform steps for this.
- The main dataset used is `wordsforthewise/lending-club` on Kaggle. Use the notebook to download and extract the CSV(s) into `data/`.

## How to run

1. Download dataset (open `dataset_download.ipynb` in Jupyter and run the cells) or run equivalent scripts to place files into `data/`.
2. Train baseline default-risk model (example):

```powershell
python feature_engineering_and_model_training.py
# or
python train_default_model.py
```

3. Train RL agent (after baseline models and environment are ready):

```powershell
python train_agent.py
```

4. Evaluate agent:

```powershell
python test_agent.py
```

## GPU (XGBoost) notes

- To use GPU acceleration with XGBoost, you'll need an NVIDIA GPU and CUDA toolkit (11.x or compatible). After installing CUDA, reinstall/upgrade XGBoost so it detects GPU support:

```powershell
pip install --upgrade xgboost
```

- In the training scripts, XGBoost is configured to use `tree_method='gpu_hist'` when available. If CUDA/GPU is not present, the script falls back to CPU training.

## Reproducibility & large files

- Large files such as CSV datasets and trained `.joblib` models are intentionally excluded from the repository via `.gitignore`.
- If you previously committed large artifacts and want to remove them from history, consider using `git filter-repo` (advanced). Contact me for help.

## Contributing

- Add issues and pull requests on GitHub. Keep data and secrets out of commits (do not add `kaggle.json`).

## License

This project is released under the MIT License — see `LICENSE`.

## Contact

If you need help running the project or want a polished README with examples and screenshots, I can expand this further.
# Personalized Loan Offers Using RL\n\nThis repo contains code for preprocessing, training baseline models (XGBoost), and prototype RL agents to personalize loan offers. See the notebooks and scripts for details.
