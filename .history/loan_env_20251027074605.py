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

        # --- 1. Load v2 Models and Data ---
        try:
            self.default_model = joblib.load('default_model_v2.joblib')
            self.acceptance_model = joblib.load('acceptance_model_v2.joblib')
            self.features_list = joblib.load('model_features_v2.joblib')
        except FileNotFoundError:
            print("Error: Model files not found. Please run (v2) training scripts first.")
            raise

        # Load raw customer data for sampling
        df = pd.read_csv(data_file, low_memory=False)
        df = df[df['loan_status'].isin(['Fully Paid', 'Charged Off'])].copy()
        
        # Must apply the *same* feature engineering
        df['term'] = df['term'].str.strip().str.replace(' months', '').astype(float)
        df['emp_length'] = df['emp_length'].str.replace(r'< 1 year', '0 years', regex=True)
        df['emp_length'] = df['emp_length'].str.replace(r'10\+ years', '10 years', regex=True)
        df['emp_length'] = df['emp_length'].str.replace(r' years?', '', regex=True)
        df['emp_length'] = pd.to_numeric(df['emp_length'], errors='coerce')

        # We keep the raw data, as models expect this format
        self.customer_data = df[self.features_list].dropna()
        
        print(f"Environment initialized with {len(self.customer_data)} customer profiles.")

        # --- 2. Define Action Space (Same) ---
        self.action_rates = np.linspace(5.0, 23.0, 10) 
        self.action_space = spaces.Discrete(len(self.action_rates))
        print(f"Action space: {len(self.action_rates)} rates: {self.action_rates}")

        # --- 3. Define Observation Space (Numeric-Only) ---
        # The agent *sees* only the numeric features for a simpler state space
        self.numeric_obs_features = [
            'loan_amnt', 'annual_inc', 'dti', 'fico_range_low', 'term', 'emp_length'
        ]
        
        lows = self.customer_data[self.numeric_obs_features].min().values
        highs = self.customer_data[self.numeric_obs_features].max().values
        highs = np.nan_to_num(highs, posinf=5_000_000)
        lows = np.nan_to_num(lows, neginf=0)
        
        self.observation_space = spaces.Box(
            low=lows.astype(np.float32), 
            high=highs.astype(np.float32),
            shape=(len(self.numeric_obs_features),), 
            dtype=np.float32
        )
        print(f"Observation space (numeric-only): {self.observation_space.shape}")
        
        self.current_customer_series = None
        self.current_customer_obs = None

    def _get_obs(self):
        return self.current_customer_obs.astype(np.float32)

    def _get_info(self):
        return {"customer_details": self.current_customer_series[self.numeric_obs_features].to_dict()}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Sample a random customer (full raw data)
        self.current_customer_series = self.customer_data.sample(n=1, random_state=self.np_random).iloc[0]
        
        # Prepare the observation (numeric part only)
        self.current_customer_obs = self.current_customer_series[self.numeric_obs_features].values
        
        if self.render_mode == "human":
            print(f"\n--- New Customer --- \n{self.current_customer_series[self.numeric_obs_features]}")

        return self._get_obs(), self._get_info()

    def step(self, action):
        offered_rate = self.action_rates[action]
        reward = 0
        terminated = True
        truncated = False
        info = self._get_info()

        # --- Build the full feature DataFrame (1 row) for our pipelines ---
        # The pipelines expect a DataFrame with all the original columns
        feature_df = pd.DataFrame([self.current_customer_series])
        feature_df['int_rate'] = offered_rate # Set the action

        # --- 1. Simulate Acceptance (using v2 pipeline) ---
        # The model pipeline handles all preprocessing internally
        prob_accept = self.acceptance_model.predict_proba(feature_df[self.features_list])[0, 1]

        if self.np_random.random() > prob_accept:
            reward = 0 
            info['outcome'] = 'Rejected'
            if self.render_mode == "human":
                print(f"  Action: Offer {offered_rate:.2f}% | Outcome: REJECTED (Prob: {prob_accept:.2f}) | Reward: {reward}")
            return self._get_obs(), reward, terminated, truncated, info

        # --- 2. Simulate Default (using v2 pipeline) ---
        prob_default = self.default_model.predict_proba(feature_df[self.features_list])[0, 1]

        if self.np_random.random() < prob_default:
            reward = -self.current_customer_series['loan_amnt']
            info['outcome'] = 'Defaulted'
            if self.render_mode == "human":
                print(f"  Action: Offer {offered_rate:.2f}% | Outcome: DEFAULTED (Prob: {prob_default:.2f}) | Reward: {reward}")
        else:
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
        annual_rate = annual_rate_pct / 100.0
        total_interest = principal * annual_rate * (term_months / 12.0)
        return total_interest

    def close(self):
        pass