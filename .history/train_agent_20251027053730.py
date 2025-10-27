import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from loan_env import LoanOfferEnv # Import your custom environment
import time
import os
import torch

print("--- Step 3: RL Agent Training (GPU Enabled) ---")

# --- 1. Check for GPU ---
if torch.cuda.is_available():
    print(f"GPU found! Using device: {torch.cuda.get_device_name(0)}")
    device = "cuda"
else:
    print("No GPU found. Defaulting to CPU.")
    device = "cpu"

# --- 2. Setup Environment ---
env = LoanOfferEnv(data_file='accepted_2007_to_2018Q4.csv')

# --- 3. Setup Model ---
# We add the 'device' parameter
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,
    buffer_size=100000,
    learning_starts=50000,
    batch_size=256,
    gamma=0.99,
    train_freq=4,
    gradient_steps=1,
    target_update_interval=1000,
    exploration_fraction=0.1,
    exploration_final_eps=0.02,
    tensorboard_log="./dqn_loan_tensorboard/",
    device=device  # <-- THIS IS THE CHANGE FOR GPU
)

# --- 4. Setup Callbacks (for saving) ---
LOG_DIR = './rl_logs/'
os.makedirs(LOG_DIR, exist_ok=True)

# Save a checkpoint every 50,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=LOG_DIR,
    name_prefix='dqn_loan_model_gpu'
)

# --- 5. Train the Agent ---
TOTAL_TIMESTEPS = 200000 # Increase this for better performance

print(f"Starting training for {TOTAL_TIMESTEPS} timesteps on {device.upper()}...")
start_time = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=checkpoint_callback,
    log_interval=100 
)
end_time = time.time()

print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes.")

# --- 6. Save Final Model ---
FINAL_MODEL_FILE = "dqn_loan_agent_final_gpu.zip"
model.save(FINAL_MODEL_FILE)

print(f"Final model saved as '{FINAL_MODEL_FILE}'")
print("--- Step 3 Complete ---")
print(f"\nTo monitor training, run: tensorboard --logdir={model.tensorboard_log}")