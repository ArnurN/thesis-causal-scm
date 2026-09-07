import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*100)
print(" ⚖️ PHASE 7: COUNTERFACTUAL BACKTESTING (SILOED VS. JOINT OPTIMIZATION)")
print("="*100)

# 1. Inputs (From Phase 6)
elasticities = {'FOODS_3_090': -111.63, 'FOODS_3_120': -37.24, 'FOODS_3_200': -0.08}
baselines = {
    'FOODS_3_090': {'price': 1.50, 'demand': 150, 'sigma': 45},
    'FOODS_3_120': {'price': 2.50, 'demand': 200, 'sigma': 30},
    'FOODS_3_200': {'price': 2.00, 'demand': 50,  'sigma': 5}
}

def get_inventory_costs(item_id, retail_price):
    if 'FOODS' in item_id:
        c, h, b = retail_price * 0.75, (retail_price * 0.75) * 0.30, retail_price * 1.50
    return c, h, b

# Helper Function: Calculate exact Profit and Safety Stock given a locked Price and Base Stock
def calculate_metrics(item_id, p, S, current_price, baseline_demand, sigma, elasticity):
    c, h, b = get_inventory_costs(item_id, p)
    
    # Causal Demand
    mu = max(1.0, baseline_demand + (elasticity * (p - current_price)))
    
    # Newsvendor Math
    z = (S - mu) / sigma
    expected_shortage = sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
    expected_sales = mu - expected_shortage
    expected_leftovers = S - expected_sales
    
    # Profit & Safety Stock
    profit = (p * expected_sales) - (c * S) - (h * expected_leftovers) - (b * expected_shortage)
    safety_stock = S - mu
    
    return profit, safety_stock, mu

# 2. Run the Comparison
results = []

for item, params in baselines.items():
    current_p = params['price']
    demand = params['demand']
    sigma = params['sigma']
    elast = elasticities[item]
    
    # -----------------------------------------------------------------
    # SCENARIO A: The "Siloed" Baseline (Marketing sets Price, Supply Chain reacts)
    # -----------------------------------------------------------------
    c_base, h_base, b_base = get_inventory_costs(item, current_p)
    # Calculate pure Critical Ratio (CR) for the old price
    cr_base = b_base / (b_base + h_base)
    # Find exact z-score for this service level
    z_base = norm.ppf(cr_base)
    # Supply Chain sets inventory based on this
    S_siloed = demand + (z_base * sigma)
    
    profit_siloed, ss_siloed, mu_siloed = calculate_metrics(item, current_p, S_siloed, current_p, demand, sigma, elast)
    
    # -----------------------------------------------------------------
    # SCENARIO B: Joint Co-Optimization (Our Phase 6C AI)
    # -----------------------------------------------------------------
    # We define the objective to let SLSQP find both P and S
    def objective(x):
        prof, _, _ = calculate_metrics(item, x[0], x[1], current_p, demand, sigma, elast)
        return -prof
    
    bounds = [(current_p * 0.5, current_p * 1.5), (demand * 0.1, demand * 3)]
    res = minimize(objective, [current_p, demand], method='SLSQP', bounds=bounds)
    
    opt_p = res.x[0]
    opt_s = res.x[1]
    
    profit_joint, ss_joint, mu_joint = calculate_metrics(item, opt_p, opt_s, current_p, demand, sigma, elast)
    
    # -----------------------------------------------------------------
    # CALCULATE THE DELTAS (The Proof)
    # -----------------------------------------------------------------
    profit_delta_pct = ((profit_joint - profit_siloed) / abs(profit_siloed)) * 100
    ss_delta_pct = ((ss_joint - ss_siloed) / abs(ss_siloed)) * 100
    
    results.append({
        'Item': item,
        'Siloed Profit': profit_siloed,
        'Joint Profit': profit_joint,
        'Profit Increase (%)': profit_delta_pct,
        'Siloed SS': ss_siloed,
        'Joint SS': ss_joint,
        'SS Reduction (%)': ss_delta_pct
    })

# 3. Print the Thesis Results Table
print(f"\n[ RESULTS ] EMPIRICAL PROOF OF BULLWHIP MITIGATION & PROFIT MAXIMIZATION\n")
print(f"{'Product ID':<15} | {'Profit (Siloed)':<17} | {'Profit (Joint)':<16} | {'Profit Growth':<15} || {'SS (Siloed)':<12} | {'SS (Joint)':<12} | {'SS Reduction (Bullwhip)'}")
print("-" * 125)

for r in results:
    print(f"{r['Item']:<15} | ${r['Siloed Profit']:<16.2f} | ${r['Joint Profit']:<15.2f} | +{r['Profit Increase (%)']:<14.1f}% || {r['Siloed SS']:<12.1f} | {r['Joint SS']:<12.1f} | {r['SS Reduction (%)']:<10.1f}%")

print("\n" + "="*100)
print(" ✅ COUNTERFACTUAL BACKTESTING COMPLETE.")
print("="*100 + "\n")
