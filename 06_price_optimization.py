import pandas as pd
import numpy as np
from scipy.optimize import minimize
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*70)
print(" 🎯 PHASE 6: PRESCRIPTIVE PRICE OPTIMIZATION (BULLWHIP MITIGATION)")
print("="*70)

# 1. Input the Causal Elasticities we discovered in Phase 5
print("\n[1] Loading Causal Elasticities (CATE) from EconML...")
elasticities = {
    'FOODS_3_090': -111.63,  # Highly Elastic
    'FOODS_3_120': -37.24,   # Moderately Elastic
    'FOODS_3_200': -0.08     # Highly Inelastic
}

# 2. Define Baseline Metrics (Averages from our historical data)
# These represent the "status quo" before our intervention
baselines = {
    'FOODS_3_090': {'price': 1.50, 'demand': 150},
    'FOODS_3_120': {'price': 2.50, 'demand': 200},
    'FOODS_3_200': {'price': 2.00, 'demand': 50}
}

print("[2] Defining Optimization Architecture...")
print("    -> Goal 1: Maximize Revenue")
print("    -> Goal 2: Minimize Demand Volatility (Stop the Bullwhip)")

# 3. The Core Optimization Function
def optimize_price(item_id, current_price, baseline_demand, elasticity):
    
    # Define the Objective Function (The "Penalty Score")
    def objective_function(proposed_price):
        # Calculate how much the price is changing
        price_change = proposed_price[0] - current_price
        
        # Calculate the counterfactual demand using our causal elasticity
        # If price goes up by $1, demand drops by the elasticity amount
        predicted_demand = baseline_demand + (elasticity * price_change)
        
        # Prevent impossible scenarios (cannot have negative demand)
        predicted_demand = max(0.1, predicted_demand)
        
        # Goal 1: Calculate Projected Revenue
        revenue = proposed_price[0] * predicted_demand
        
        # Goal 2: Calculate Volatility Penalty (Bullwhip Factor)
        # We severely penalize prices that cause massive shifts from the baseline demand
        volatility_penalty = abs(predicted_demand - baseline_demand) * 2.0 
        
        # We want to MAXIMIZE revenue, which means MINIMIZING negative revenue.
        # We add the volatility penalty so the optimizer tries to shrink both.
        total_penalty = -revenue + volatility_penalty
        
        return total_penalty

    # Set boundaries: The optimizer is only allowed to change the price by +/- 50%
    bounds = [(current_price * 0.5, current_price * 1.5)]
    
    # Start the optimizer at the current baseline price
    initial_guess = [current_price]
    
    # Run the SciPy SLSQP Optimization Algorithm
    result = minimize(
        objective_function, 
        initial_guess, 
        method='SLSQP', 
        bounds=bounds
    )
    
    optimized_price = result.x[0]
    final_demand = max(0, baseline_demand + (elasticity * (optimized_price - current_price)))
    final_revenue = optimized_price * final_demand
    
    return optimized_price, final_demand, final_revenue

# 4. Execute Optimization for Each Item
print("\n[3] Executing Optimization Engine...\n")
print(f"{'Product ID':<15} | {'Old Price':<10} | {'New Optimal Price':<18} | {'Recommendation'}")
print("-" * 75)

for item, params in baselines.items():
    opt_price, exp_demand, exp_rev = optimize_price(
        item_id=item,
        current_price=params['price'],
        baseline_demand=params['demand'],
        elasticity=elasticities[item]
    )
    
    old_price = params['price']
    
    if opt_price > old_price + 0.05:
        action = "INCREASE PRICE"
    elif opt_price < old_price - 0.05:
        action = "DECREASE PRICE"
    else:
        action = "HOLD STEADY"
        
    print(f"{item:<15} | ${old_price:<9.2f} | ${opt_price:<17.2f} | {action}")

print("\n" + "="*70)
print(" ✅ OPTIMIZATION COMPLETE: Causal Policy successfully generated.")
print("="*70 + "\n")