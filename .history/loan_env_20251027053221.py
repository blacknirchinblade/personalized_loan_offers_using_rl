import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
import joblib

class LoanOfferEnv(gym.Env):
    metadata = {'render_modes': ['human']}

    def __init__(self, data_file='accepted_2007_to_2018Q4.csv', render_mode=None):
        super(LoanOfferEnv, self).__init__()

        self.render_mode = render_mode

        # --- 1. Load Models and Data ---
        try:
            self.default_model = joblib.load('default_model.joblib')
            self.acceptance_model = joblib.load('acceptance_model.joblib')
            self.features_list = joblib.load('model_features.joblib')
        except FileNotFoundError:
            print("Error: Model files not found. Please run training scripts first.")
            raise

        # Load customer data for sampling
        df = pd.read_csv(data_file, low_memory=False)
        df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
        df['term'] = df['term'].str.strip().str.replace(' months', '').astype(float)
        self.customer_data = df[self.features_list].dropna()
        
        # We need the non-int_rate features for the observation
        self.observation_features = [f for f in self.features_list if f != 'int_rate']

        print(f"Environment initialized with {len(self.customer_data)} customer profiles.")

        # --- 2. Define Action Space (Discrete Interest Rates) ---
        # 10 possible interest rates, e.g., 5%, 7%, 9%, ... 23%
        self.action_rates = np.linspace(5.0, 23.0, 10) 
        self.action_space = spaces.Discrete(len(self.action_rates))
        print(f"Action space: {len(self.action_rates)} rates: {self.action_rates}")

        # --- 3. Define Observation Space (Customer Features) ---
        # Features: 'loan_amnt', 'annual_inc', 'dti', 'fico_range_low', 'term'
        # We need to define reasonable min/max boundaries for scaling
        lows = self.customer_data[self.observation_features].min().values
        highs = self.customer_data[self.observation_features].max().values
        
        # A quick fix for any potential -inf/+inf from .min()/.max() on large data
        highs = np.nan_to_num(highs, posinf=5_000_000) # e.g. max annual_inc
        lows = np.nan_to_num(lows, neginf=0)
        
        self.observation_space = spaces.Box(
            low=lows.astype(np.float32), 
            high=highs.astype(np.float32),
            shape=(len(self.observation_features),), 
            dtype=np.float32
        )
        print(f"Observation space shape: {self.observation_space.shape}")

        self.current_customer_series = None
        self.current_customer_obs = None

    def _get_obs(self):
        return self.current_customer_obs.astype(np.float32)

    def _get_info(self):
        return {"customer_details": self.current_customer_series.to_dict()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed) # Required for gymnasium
        
        # Sample a random customer
        self.current_customer_series = self.customer_data.sample(n=1, random_state=self.np_random).iloc[0]
        
        # Prepare the observation (all features *except* int_rate)
        self.current_customer_obs = self.current_customer_series[self.observation_features].values
        
        if self.render_mode == "human":
            print(f"\n--- New Customer --- \n{self.current_customer_series[self.observation_features]}")

        return self._get_obs(), self._get_info()

    def step(self, action):
        offered_rate = self.action_rates[action]
        reward = 0
        terminated = True # This is a one-step (episodic) environment
        truncated = False
        info = self._get_info()

        # --- 1. Simulate Acceptance ---
        # Re-build the feature vector for the acceptance model
        accept_features = self.current_customer_series.copy()
        accept_features['int_rate'] = offered_rate
        
        # Get acceptance probability
        prob_accept = self.acceptance_model.predict_proba([accept_features[self.features_list]])[0, 1]

        if self.np_random.random() > prob_accept:
            # SCENARIO 1: Customer Rejected
            reward = 0 # No profit, no loss
            info['outcome'] = 'Rejected'
            if self.render_mode == "human":
                print(f"  Action: Offer {offered_rate:.2f}% | Outcome: REJECTED (Prob: {prob_accept:.2f}) | Reward: {reward}")
            return self._get_obs(), reward, terminated, truncated, info

        # --- 2. Simulate Default (if accepted) ---
        # SCENARIO 2: Customer Accepted
        # Re-build the feature vector for the default model
        default_features = self.current_customer_series.copy()
        default_features['int_rate'] = offered_rate

        # Get default probability
        prob_default = self.default_model.predict_proba([default_features[self.features_list]])[0, 1]

        if self.np_random.random() < prob_default:
            # SCENARIO 2a: Accepted, but Defaulted
            # Reward is the loss of the entire loan principal
            reward = -self.current_customer_series['loan_amnt']
            info['outcome'] = 'Defaulted'
            if self.render_mode == "human":
                print(f"  Action: Offer {offered_rate:.2f}% | Outcome: DEFAULTED (Prob: {prob_default:.2f}) | Reward: {reward}")
        else:
            # SCENARIO 2b: Accepted and Fully Paid
            # Reward is the total profit from interest
            reward = self._calculate_profit(
                self.current_customer_series['loan_amnt'],
                offered_rate,
                self.current_customer_series['term']
            )
            info['outcome'] = 'Fully Paid'
            if self.render_mode == "human":
                print(f"  Action: Offer {offered_rate:.2f}% | Outcome: FULLY PAID (Prob: {prob_default:.2f}) | Reward: {reward}")
        
        return self._get_obs(), reward, terminated, truncated, info

    def _calculate_profit(self, principal, annual_rate_pct, term_months):
        # Simple interest calculation for profit
        # P * (r/100) * (t/12)
        annual_rate = annual_rate_pct / 100.0
        total_interest = principal * annual_rate * (term_months / 12.0)
        return total_interest

    def close(self):
        pass