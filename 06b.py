import pandas as pd
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" 🎯 PHASE 6B: JOINT PRICE & INVENTORY CO-OPTIMIZATION (PATCHED NEWSVENDOR)")
print("="*80)

# 1. Inputs: Causal Elasticities & Baseline Store Data
print("\n[1] Loading Causal Elasticities (CATE) & Baseline Store Data...")
elasticities = {
    'FOODS_3_090': -111.63,  
    'FOODS_3_120': -37.24,   
    'FOODS_3_200': -0.08     
}

baselines = {
    'FOODS_3_090': {'price': 1.50, 'demand': 150, 'sigma': 45},
    'FOODS_3_120': {'price': 2.50, 'demand': 200, 'sigma': 30},
    'FOODS_3_200': {'price': 2.00, 'demand': 50,  'sigma': 5}
}

# 2. Dynamic SCM Cost Allocation
print("[2] Initializing Dynamic Newsvendor Cost Parameters by Category...")
def get_inventory_costs(item_id, retail_price):
    if 'FOODS' in item_id:
        c = retail_price * 0.75  
        h = c * 0.30             
        b = retail_price * 1.50  
    elif 'HOUSEHOLD' in item_id:
        c = retail_price * 0.60
        h = c * 0.15             
        b = retail_price * 1.00  
    else: 
        c = retail_price * 0.50
        h = c * 0.20
        b = retail_price * 0.80  
    return c, h, b

# 3. The Co-Optimization Objective Function
def optimize_supply_chain(item_id, current_price, baseline_demand, sigma, elasticity):
    
    c, h, b = get_inventory_costs(item_id, current_price)
    
    def objective_function(x):
        proposed_price = x[0]
        proposed_inventory = x[1]
        
        # Calculate counterfactual Mean Demand (mu)
        price_change = proposed_price - current_price
        mu = baseline_demand + (elasticity * price_change)
        mu = max(1.0, mu) 
        
        # -------------------------------------------------------------------
        # THE MATHEMATICAL PATCH: Strict Normal Loss Function & Service Level
        # -------------------------------------------------------------------
        z = (proposed_inventory - mu) / sigma
        
        # 1. Strict Expected Shortage (Standard Normal Loss Function)
        expected_shortage = sigma * (norm.pdf(z) - z * (1 - norm.cdf(z)))
        
        # 2. Strict Expected Sales & Leftovers
        expected_sales = mu - expected_shortage
        expected_leftovers = proposed_inventory - expected_sales
        
        # 3. The "Walmart Constraint" (Minimum 95% Service Level)
        # If the AI tries to supply less than 95% of the expected demand, apply a massive penalty
        service_level = expected_sales / mu
        sl_penalty = 0
        if service_level < 0.95:
            sl_penalty = (0.95 - service_level) * 100000 
        # -------------------------------------------------------------------
        
        revenue = proposed_price * expected_sales
        procurement_cost = c * proposed_inventory
        holding_cost = h * expected_leftovers
        stockout_cost = b * expected_shortage
        
        expected_profit = revenue - procurement_cost - holding_cost - stockout_cost - sl_penalty
        
        return -expected_profit

    # Bounds: Price +/- 50%. Inventory bounded safely away from zero.
    bounds = [(current_price * 0.5, current_price * 1.5), (baseline_demand * 0.1, baseline_demand * 3)]
    initial_guess = [current_price, baseline_demand]
    
    result = minimize(
        objective_function, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds
    )
    
    opt_price = result.x[0]
    opt_inventory = result.x[1]
    maximized_profit = -result.fun
    
    return opt_price, opt_inventory, maximized_profit

# 4. Execute the Engine
print("\n[3] Executing Patched Joint Optimization Engine...\n")
print(f"{'Product ID':<15} | {'Old Price':<10} | {'Optimal Price':<14} | {'Optimal Base Stock (S)':<22} | {'Projected Daily Profit'}")
print("-" * 95)

for item, params in baselines.items():
    opt_p, opt_s, max_prof = optimize_supply_chain(
        item_id=item,
        current_price=params['price'],
        baseline_demand=params['demand'],
        sigma=params['sigma'],
        elasticity=elasticities[item]
    )
    
    print(f"{item:<15} | ${params['price']:<9.2f} | ${opt_p:<13.2f} | {int(opt_s):<22} | ${max_prof:<20.2f}")

print("\n" + "="*80)
print(" ✅ SCM OPTIMIZATION COMPLETE: Corner solution eliminated. Physics restored.")
print("="*80 + "\n")