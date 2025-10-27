import gymnasium as gym
from stable_baselines3 import DQN
from loan_env import LoanOfferEnv # Import your custom environment
import time

print("--- Step 4: Test Trained RL Agent ---")

MODEL_FILE = "dqn_loan_agent_final.zip"
NUM_TEST_EPISODES = 10

# --- 1. Load Environment and Model ---
# Set render_mode="human" to see the printouts from the env
print("Loading environment in 'human' render mode...")
env = LoanOfferEnv(data_file='accepted_2007_to_2018Q4.csv', render_mode="human")

print(f"Loading trained model from {MODEL_FILE}...")
try:
    model = DQN.load(MODEL_FILE, env=env)
except FileNotFoundError:
    print(f"Error: Model file '{MODEL_FILE}' not found.")
    print("Please run 'train_agent.py' first.")
    exit()

print("Model loaded successfully.")

# --- 2. Test the Agent's Policy ---
print(f"\n--- Testing agent on {NUM_TEST_EPISODES} random customers ---")
total_reward = 0
for i in range(NUM_TEST_EPISODES):
    obs, info = env.reset()
    
    # Use deterministic=True to get the agent's "best" action
    action, _states = model.predict(obs, deterministic=True)
    
    # Pass the action to the environment
    obs, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward

print(f"\n--- Test Complete ---")
print(f"Total reward over {NUM_TEST_EPISODES} customers: {total_reward:.2f}")
print(f"Average reward per customer: {total_reward / NUM_TEST_EPISODES:.2f}")