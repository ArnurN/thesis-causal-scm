import pandas as pd
import numpy as np
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
import warnings

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

print("\n" + "="*70)
print(" 🛡️ PHASE 5B: CAUSAL ROBUSTNESS & REFUTATION TESTING")
print("="*70)

# 1. Load Data and Define Variables
print("\n[1] Loading Enhanced Dataset...")
df = pd.read_csv('mvt_master.csv')
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek
df = pd.get_dummies(df, columns=['item_id'], drop_first=True)
item_cols = [col for col in df.columns if 'item_id_' in col]
Y = df['units_sold']
X = df[item_cols] 
W = df[['month', 'day_of_week', 'discount_depth_percentage', 'short_term_volatility_index']]
T = df['sell_price']

# Define the base DML model architecture
def run_dml(Y, T, X, W, random_state=42):
    est = LinearDML(
        model_y=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state),
        model_t=RandomForestRegressor(n_estimators=100, max_depth=5, random_state=random_state),
        random_state=random_state,
        discrete_treatment=False
    )
    est.fit(Y, T, X=X, W=W)
    return est.ate(X)

# 2. Re-Run Base Model for Reference
print("\n[2] Establishing Base Causal Effect...")
base_ate = run_dml(Y, T, X, W)
print(f"    -> Original ATE: {base_ate:.2f} units lost per $1 increase")

# 3. TEST A: The Random Confounder Test
print("\n[3] INITIATING TEST A: Random Confounder (Dummy Variable) Test")
print("    Theory: Adding random garbage data should NOT change the true elasticity.")
# Create a completely random variable (e.g., "Phases of the Moon")
np.random.seed(42)
W_dummy = W.copy()
W_dummy['random_garbage_noise'] = np.random.normal(0, 1, size=len(df))

dummy_ate = run_dml(Y, T, X, W_dummy)
print(f"    -> New ATE with random noise: {dummy_ate:.2f} units")
difference_a = abs(base_ate - dummy_ate)
if difference_a < 2.0:
    print("    ✅ PASSED: Model ignored the fake data. Elasticity is stable.")
else:
    print("    ❌ FAILED: Model is fragile and got confused by random noise.")

# 4. TEST B: The Placebo Treatment Test
print("\n[4] INITIATING TEST B: Placebo Treatment (Fake Price) Test")
print("    Theory: If we randomize prices, they no longer cause sales. Elasticity MUST drop to 0.")
# Shuffle the prices randomly so they no longer match the actual days they were sold
np.random.seed(42)
T_placebo = np.random.permutation(T)

placebo_ate = run_dml(Y, T_placebo, X, W)
print(f"    -> Placebo ATE with fake prices: {placebo_ate:.2f} units")
if abs(placebo_ate) < 5.0:
    print("    ✅ PASSED: Model successfully detected that the fake prices have no causal power.")
else:
    print("    ❌ FAILED: Model hallucinated a causal effect from random prices.")

print("\n" + "="*70)
print(" 🏁 VALIDATION COMPLETE: If both passed, your EconML logic is bulletproof.")
print("="*70 + "\n")