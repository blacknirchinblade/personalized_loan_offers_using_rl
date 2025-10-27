import gymnasium as gym
from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback
from loan_env import LoanOfferEnv # Import your custom environment
import time
import os

print("--- Step 3: RL Agent Training ---")

# --- 1. Setup Environment ---
# We use the same data file to load customer profiles
# We don't set render_mode="human" during training for speed
env = LoanOfferEnv(data_file='accepted_2007_to_2018Q4.csv')

# --- 2. Setup Model ---
# DQN is a great choice for discrete actions
# We'll use a larger buffer and some other tuned parameters
model = DQN(
    "MlpPolicy",
    env,
    verbose=1,
    learning_rate=1e-4,         # Lower learning rate for more stable learning
    buffer_size=100000,         # Size of the replay buffer
    learning_starts=50000,      # Start learning after 50k steps
    batch_size=256,             # Larger batch size
    gamma=0.99,                 # Discount factor
    train_freq=4,               # Train the model every 4 steps
    gradient_steps=1,
    target_update_interval=1000, # Update the target network
    exploration_fraction=0.1,   # Explore for the first 10% of steps
    exploration_final_eps=0.02,
    tensorboard_log="./dqn_loan_tensorboard/"
)

# --- 3. Setup Callbacks (for saving) ---
LOG_DIR = './rl_logs/'
os.makedirs(LOG_DIR, exist_ok=True)

# Save a checkpoint every 50,000 steps
checkpoint_callback = CheckpointCallback(
    save_freq=50000,
    save_path=LOG_DIR,
    name_prefix='dqn_loan_model'
)

# --- 4. Train the Agent ---
TOTAL_TIMESTEPS = 200000 # Increase this for better performance (e.g., 500k, 1M)

print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
start_time = time.time()
model.learn(
    total_timesteps=TOTAL_TIMESTEPS, 
    callback=checkpoint_callback,
    log_interval=100 # Log stats every 100 episodes
)
end_time = time.time()

print(f"Training finished in {(end_time - start_time) / 60:.2f} minutes.")

# --- 5. Save Final Model ---
FINAL_MODEL_FILE = "dqn_loan_agent_final.zip"
model.save(FINAL_MODEL_FILE)

print(f"Final model saved as '{FINAL_MODEL_FILE}'")
print("--- Step 3 Complete ---")
print(f"\nTo monitor training, run: tensorboard --logdir={model.tensorboard_log}")