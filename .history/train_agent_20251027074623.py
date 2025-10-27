import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from loan_env import LoanOfferEnv # Import your v2 custom environment
import time
import os
import torch

print("--- Step 3 (v2): RL Agent Training (Upgraded) ---")

# --- 1. Check for GPU ---
if torch.cuda.is_available():
    print(f"GPU found! Using device: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("No GPU found. Defaulting to CPU.")
    device = "cpu"

# --- 2. Setup Environment ---
env = LoanOfferEnv(data_file='accepted_2007_to_2018Q4.csv')

# --- 3. Setup Model (Upgraded) ---
# Define a larger network for our "smarter" agent
policy_kwargs = dict(
    net_arch=[128, 128] # Default is [64, 64]
)

model = DQN(
    "MlpPolicy",
    env,
    policy_kwargs=policy_kwargs,    # <-- Use larger network
    verbose=1,
    learning_rate=1e-4,
    buffer_size=200000,             # <-- Increased buffer size
    learning_starts=50000,
    batch_size=256,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.25,      # <-- Increased exploration
    exploration_final_eps=0.02,
    tensorboard_log="./dqn_loan_tensorboard_v2/",
    device=device
)

# --- 4. Setup Callbacks (for saving) ---
LOG_DIR = './rl_logs_v2/'
os.makedirs(LOG_DIR, exist_ok=True)

# Save a checkpoint every 100,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=100000,
    save_path=LOG_DIR,
    name_prefix='dqn_loan_model_v2'
)

# --- 5. Train the Agent (Longer) ---
TOTAL_TIMESTEPS = 1000000 # <-- INCREASED TRAINING TIME

print(f"Starting training for {TOTAL_TIMESTEPS} timesteps on {device.upper()}...")
start_time = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=checkpoint_callback,
    log_interval=1000 # Log stats every 1000 episodes
)
end_time = time.time()

print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes.")

# --- 6. Save Final Model ---
FINAL_MODEL_FILE = "dqn_loan_agent_final_v2.zip"
model.save(FINAL_MODEL_FILE)

print(f"Final model saved as '{FINAL_MODEL_FILE}'")
print("--- Step 3 (v2) Complete ---")
print(f"\nTo monitor training, run: tensorboard --logdir={model.tensorboard_log}")