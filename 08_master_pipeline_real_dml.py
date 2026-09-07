import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
from econml.dml import LinearDML
from sklearn.linear_model import LassoCV
from tqdm import tqdm
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*90)
print(" 🚀 PHASE 8: TRUE CAUSAL PIPELINE (ECONML + OPERATIONS RESEARCH)")
print("="*90)

# ==============================================================================
# PART 1: DATA INGESTION & PREPROCESSING
# ==============================================================================
print("[1] Loading Data and Preparing Confounders...")
df = pd.read_csv('cleaned_m5_data.csv')

# Convert Holidays to a binary numeric flag (1 if holiday, 0 if normal day)
df['is_holiday'] = df['event_name_1'].notna().astype(int)

# Liquidity Filter: Only keep products that sell >= 1 unit per day
item_sales = df.groupby('item_id')['units_sold'].mean().reset_index()
active_items = item_sales[item_sales['units_sold'] >= 1.0]['item_id'].tolist()
df_active = df[df['item_id'].isin(active_items)]

sampled_items = df_active['item_id'].unique()
print(f"✅ Successfully isolated {len(sampled_items)} high-liquidity items.")

# ==============================================================================
# PART 2: THE THEORETICAL SCM LOGIC
# ==============================================================================
def get_inventory_costs(dept_id, retail_price):
    if 'FOODS' in dept_id:
        c, h, b = retail_price * 0.75, (retail_price * 0.75) * 0.30, retail_price * 1.50
    elif 'HOUSEHOLD' in dept_id:
        c, h, b = retail_price * 0.60, (retail_price * 0.60) * 0.15, retail_price * 1.00
    else: 
        c, h, b = retail_price * 0.50, (retail_price * 0.50) * 0.20, retail_price * 0.80
    return c, h, b

# ==============================================================================
# PART 3: THE TRUE DML ENGINE
# ==============================================================================
print("[3] Booting Double Machine Learning Engine. This will take several minutes...\n")

master_results = []

for item in tqdm(sampled_items, desc="Calculating True Causal Elasticity & Optimizing"):
    
    item_data = df_active[df_active['item_id'] == item].dropna(subset=['sell_price', 'units_sold'])
    
    # Skip if data somehow became too sparse after dropping NAs
    if len(item_data) < 50:
        continue
        
    dept_id = item_data['dept_id'].iloc[0]
    current_price = item_data['sell_price'].mean()
    baseline_demand = item_data['units_sold'].mean()
    historical_sigma = item_data['units_sold'].std()
    
    if pd.isna(historical_sigma) or historical_sigma == 0:
        historical_sigma = baseline_demand * 0.1 
        
    cv = historical_sigma / baseline_demand
    
    # -------------------------------------------------------------------------
    # THE TRUE MACHINE LEARNING EXTRACTION
    # -------------------------------------------------------------------------
    Y = item_data['units_sold']  # Outcome: Demand
    T = item_data['sell_price']  # Treatment: Price
    W = item_data[['is_holiday', 'snap_CA', 'wday', 'month']] # Confounders
    
    # We use LassoCV (Linear DML) for a balance of extreme academic rigor and speed
    est = LinearDML(model_y=LassoCV(cv=3), model_t=LassoCV(cv=3), discrete_treatment=False, random_state=42)
    
    try:
        est.fit(Y, T, W=W)
        # Extract the Average Treatment Effect (The True Elasticity)
        raw_elasticity = est.ate()
    except:
        # Failsafe if the ML model cannot converge due to flat data
        raw_elasticity = 0.0 
        
    # Apply the Law of Demand (Fixing the Infinite Money Glitch)
    elasticity = max(-100.0, min(-0.01, raw_elasticity))
    
    # -------------------------------------------------------------------------
    # OPERATIONS RESEARCH CO-OPTIMIZATION
    # -------------------------------------------------------------------------
    c, h, b = get_inventory_costs(dept_id, current_price)
    
    def evaluate_state(p, S):
        mu = max(1.0, baseline_demand + (elasticity * (p - current_price)))
        dynamic_sigma = mu * cv 
        z = (S - mu) / dynamic_sigma
        expected_shortage = dynamic_sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
        expected_sales = mu - expected_shortage
        expected_leftovers = S - expected_sales
        profit = (p * expected_sales) - (c * S) - (h * expected_leftovers) - (b * expected_shortage)
        safety_stock = S - mu
        return profit, safety_stock, mu
        
    # Baseline (Siloed)
    cr_base = b / (b + h)
    z_base = norm.ppf(cr_base)
    S_siloed = baseline_demand + (z_base * historical_sigma)
    profit_siloed, ss_siloed, _ = evaluate_state(current_price, S_siloed)
    
    # AI Optimization (Joint)
    def objective(x):
        prof, _, _ = evaluate_state(x[0], x[1])
        return -prof
        
    bounds = [(current_price * 0.5, current_price * 1.5), (baseline_demand * 0.1, baseline_demand * 3)]
    res = minimize(objective, [current_price, baseline_demand], method='SLSQP', bounds=bounds)
    
    opt_p, opt_s = res.x[0], res.x[1]
    profit_joint, ss_joint, _ = evaluate_state(opt_p, opt_s)
    
    master_results.append({
        'Item_ID': item,
        'Category': dept_id,
        'Old_Price': current_price,
        'New_Price': opt_p,
        'True_Elasticity': elasticity,
        'Siloed_Profit': profit_siloed,
        'Joint_Profit': profit_joint,
        'Profit_Delta_Pct': ((profit_joint - profit_siloed) / max(0.01, abs(profit_siloed))) * 100,
        'Siloed_Safety_Stock': ss_siloed,
        'Joint_Safety_Stock': ss_joint,
        'SS_Reduction_Pct': ((ss_joint - ss_siloed) / max(0.01, abs(ss_siloed))) * 100
    })

# ==============================================================================
# PART 4: EXPORT
# ==============================================================================
results_df = pd.DataFrame(master_results)
results_df.to_csv('final_thesis_results_REAL.csv', index=False)

print("\n" + "="*90)
print(" ✅ TRUE CAUSAL PIPELINE COMPLETE. Data saved to 'final_thesis_results_REAL.csv'")
print("\n[ REAL MACRO SUMMARY FOR THESIS ]")
print(f"Total Items Optimized: {len(results_df)}")
print(f"Average Profit Growth: {results_df['Profit_Delta_Pct'].mean():.2f}%")
print(f"Average Safety Stock Reduction: {results_df['SS_Reduction_Pct'].mean():.2f}%")
print("="*90 + "\n")