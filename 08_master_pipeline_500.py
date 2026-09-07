import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings
from tqdm import tqdm # Progress bar for the 500 items

warnings.filterwarnings('ignore')

print("\n" + "="*90)
print(" 🚀 PHASE 8: THE MASTER PIPELINE (500-ITEM EXPANSION)")
print("="*90)

# ==============================================================================
# PART 1: DATA INGESTION & SAMPLING
# ==============================================================================
print("[1] Loading Data and Applying Liquidity Filter...")

try:
    # Replace with your actual file path
    df = pd.read_csv('cleaned_m5_data.csv')
except FileNotFoundError:
    print("❌ ERROR: 'cleaned_m5_data.csv' not found. Ensure the Phase 2 dataset is in the directory.")
    exit()

# 1. Liquidity Filter: Only keep products that sell at least 1 unit per day on average
item_sales = df.groupby('item_id')['units_sold'].mean().reset_index()
active_items = item_sales[item_sales['units_sold'] >= 1.0]['item_id'].tolist()
df_active = df[df['item_id'].isin(active_items)]

# 2. Finalizing Active Items (Sampling was already done in Phase 1)
print("[2] Finalizing Active Items...")
sampled_df = df_active.copy()

sampled_items = sampled_df['item_id'].unique()
print(f"✅ Successfully sampled {len(sampled_items)} active items.")

# ==============================================================================
# PART 2: THE THEORETICAL SCM LOGIC
# ==============================================================================
def get_inventory_costs(dept_id, retail_price):
    if 'FOODS' in dept_id:
        c, h, b = retail_price * 0.75, (retail_price * 0.75) * 0.30, retail_price * 1.50
    elif 'HOUSEHOLD' in dept_id:
        c, h, b = retail_price * 0.60, (retail_price * 0.60) * 0.15, retail_price * 1.00
    else: # HOBBIES
        c, h, b = retail_price * 0.50, (retail_price * 0.50) * 0.20, retail_price * 0.80
    return c, h, b

# ==============================================================================
# PART 3: THE ENGINE (DML + OR + COUNTERFACTUAL BACKTESTING)
# ==============================================================================
print("[3] Initializing DML & Co-Optimization Engine...\n")

# For the master script, we simulate the DML elasticity extraction for speed,
# applying the Law of Demand constraint. In your actual thesis, you will 
# merge this loop with the EconML output table from Phase 5.
np.random.seed(42) 

master_results = []

for item in tqdm(sampled_items, desc="Optimizing Supply Chain"):
    
    # A. Empirical Baseline Data for this specific item
    item_data = sampled_df[sampled_df['item_id'] == item]
    dept_id = item_data['dept_id'].iloc[0]
    
    current_price = item_data['sell_price'].mean()
    baseline_demand = item_data['units_sold'].mean()
    historical_sigma = item_data['units_sold'].std()
    
    # Handle mathematical edge cases (Zero variance)
    if pd.isna(historical_sigma) or historical_sigma == 0:
        historical_sigma = baseline_demand * 0.1 
        
    # Constant Coefficient of Variation (CV) Assumption
    cv = historical_sigma / baseline_demand
    
    # B. The Causal Elasticity (Simulated from EconML Phase)
    # Applying the "Law of Demand" constraint: Force to be negative
    raw_elasticity = np.random.normal(-5.0, 15.0) 
    elasticity = min(-0.01, raw_elasticity) # Never allow positive elasticity
    
    c, h, b = get_inventory_costs(dept_id, current_price)
    
    # C. Helper: Evaluate Financial Physics
    def evaluate_state(p, S):
        mu = max(1.0, baseline_demand + (elasticity * (p - current_price)))
        dynamic_sigma = mu * cv # Dynamic Volatility Scaling
        
        z = (S - mu) / dynamic_sigma
        expected_shortage = dynamic_sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
        expected_sales = mu - expected_shortage
        expected_leftovers = S - expected_sales
        
        profit = (p * expected_sales) - (c * S) - (h * expected_leftovers) - (b * expected_shortage)
        safety_stock = S - mu
        return profit, safety_stock, mu
        
    # D. Scenario 1: Siloed Baseline (Historical)
    cr_base = b / (b + h)
    z_base = norm.ppf(cr_base)
    S_siloed = baseline_demand + (z_base * historical_sigma)
    profit_siloed, ss_siloed, _ = evaluate_state(current_price, S_siloed)
    
    # E. Scenario 2: Joint Co-Optimization (AI)
    def objective(x):
        prof, _, _ = evaluate_state(x[0], x[1])
        return -prof
        
    bounds = [(current_price * 0.5, current_price * 1.5), (baseline_demand * 0.1, baseline_demand * 3)]
    res = minimize(objective, [current_price, baseline_demand], method='SLSQP', bounds=bounds)
    
    opt_p = res.x[0]
    opt_s = res.x[1]
    profit_joint, ss_joint, opt_mu = evaluate_state(opt_p, opt_s)
    
    # F. Record Results
    master_results.append({
        'Item_ID': item,
        'Category': dept_id,
        'Old_Price': current_price,
        'New_Price': opt_p,
        'Elasticity': elasticity,
        'Siloed_Profit': profit_siloed,
        'Joint_Profit': profit_joint,
        'Profit_Delta_Pct': ((profit_joint - profit_siloed) / max(0.01, abs(profit_siloed))) * 100,
        'Siloed_Safety_Stock': ss_siloed,
        'Joint_Safety_Stock': ss_joint,
        'SS_Reduction_Pct': ((ss_joint - ss_siloed) / max(0.01, abs(ss_siloed))) * 100
    })

# ==============================================================================
# PART 4: AGGREGATE RESULTS & EXPORT
# ==============================================================================
results_df = pd.DataFrame(master_results)
results_df.to_csv('final_thesis_results.csv', index=False)

print("\n" + "="*90)
print(" ✅ PIPELINE COMPLETE. Data saved to 'final_thesis_results.csv'")
print("\n[ MACRO SUMMARY FOR THESIS ]")
print(f"Total Items Optimized: {len(results_df)}")
print(f"Average Profit Growth: {results_df['Profit_Delta_Pct'].mean():.2f}%")
print(f"Average Safety Stock Reduction (Bullwhip Mitigation): {results_df['SS_Reduction_Pct'].mean():.2f}%")
print("="*90 + "\n")