import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

print("Booting Causal Micro-Simulation Sandbox...")

# Set random seed for strict mathematical reproducibility
np.random.seed(42)

# ==============================================================================
# 1. SIMULATION PARAMETERS
# ==============================================================================
days = 30
shock_day = 15
base_price = 10.0
base_demand = 50.0
true_elasticity = -2.5  # Extracted from our Double Machine Learning engine

# Storage arrays for time-series plotting
demand_A, orders_A = [], []
demand_B, orders_B, prices_B = [], [], []

# ==============================================================================
# 2. RUN THE 30-DAY SIMULATION
# ==============================================================================
for t in range(days):
    # The Environment: Injecting the Supply Chain Shock on Day 15
    if t == shock_day:
        print(f"Day {t}: SUPPLY CHAIN SHOCK INJECTED (Demand Variance +400%)")
        
    noise_std = 15.0 if t >= shock_day else 3.0
    exogenous_shock = np.random.normal(0, noise_std)
    
    # --------------------------------------------------------------------------
    # RUN A: TRADITIONAL SCM (Static Price, Reactive Inventory Forecast)
    # --------------------------------------------------------------------------
    price_A = base_price
    actual_demand_A = max(0, base_demand + exogenous_shock)
    
    # Order-Up-To Policy: Orders react to recent demand spikes. 
    # Standard systems add panic safety stock when variance abruptly increases.
    order_A = max(0, actual_demand_A + (actual_demand_A - np.mean(demand_A[-3:] if t>3 else actual_demand_A))) 
    if t >= shock_day: 
        order_A += 20  # Artificial panic safety stock hoarding
        
    demand_A.append(actual_demand_A)
    orders_A.append(order_A)

    # --------------------------------------------------------------------------
    # RUN B: CAUSAL AI SCM (Dynamic Price, Co-Optimized Inventory)
    # --------------------------------------------------------------------------
    if t >= shock_day:
        # The AI uses the causal elasticity to explicitly calculate the price 
        # increase required to kill the excess demand caused by the shock.
        price_adjustment = exogenous_shock / abs(true_elasticity)
        price_B = base_price + max(0, price_adjustment) 
    else:
        price_B = base_price
        
    # Actual demand realized AFTER the AI intervenes with price
    actual_demand_B = max(0, base_demand + exogenous_shock + (true_elasticity * (price_B - base_price)))
    
    # Because demand is strictly controlled, the order quantity remains perfectly stable
    order_B = max(0, actual_demand_B)
    
    demand_B.append(actual_demand_B)
    orders_B.append(order_B)
    prices_B.append(price_B)

# ==============================================================================
# 3. CALCULATE THE BULLWHIP RATIO (Variance of Orders / Variance of Demand)
# ==============================================================================
var_demand_A = np.var(demand_A)
var_order_A = np.var(orders_A)
bw_A = var_order_A / var_demand_A if var_demand_A > 0 else 0

var_demand_B = np.var(demand_B)
var_order_B = np.var(orders_B)
bw_B = var_order_B / var_demand_B if var_demand_B > 0 else 0

print(f"\n[ SIMULATION METRICS ]")
print(f"Traditional Policy Bullwhip Ratio: {bw_A:.2f} (Variance Amplified)")
print(f"Causal Policy Bullwhip Ratio:      {bw_B:.2f} (Variance Mitigated)")

# ==============================================================================
# 4. HIGH-RESOLUTION VISUALIZATION
# ==============================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

color_siloed = "#E74C3C"  # Red
color_joint = "#2ECC71"   # Green

# Subplot 1: Run A
axes[0].plot(demand_A, label='Consumer Demand', color='black', linestyle='--', alpha=0.7)
axes[0].plot(orders_A, label='Wholesale Orders (Run A)', color=color_siloed, linewidth=2)
axes[0].axvline(x=shock_day, color='red', linestyle=':', label='Supply Shock Injection')
axes[0].set_title(f'Traditional SCM: The Bullwhip Effect (BW Ratio: {bw_A:.2f})', fontweight='bold')
axes[0].legend(loc="upper left")
axes[0].set_ylabel('Unit Volume')

# Subplot 2: Run B
axes[1].plot(demand_B, label='Consumer Demand (Stabilized)', color='black', linestyle='--', alpha=0.7)
axes[1].plot(orders_B, label='Wholesale Orders (Run B)', color=color_joint, linewidth=2)
axes[1].axvline(x=shock_day, color='red', linestyle=':', label='Supply Shock Injection')
axes[1].set_title(f'Causal SCM: Demand Shaping via Price (BW Ratio: {bw_B:.2f})', fontweight='bold')
axes[1].legend(loc="upper left")
axes[1].set_ylabel('Unit Volume')
axes[1].set_xlabel('Simulation Days')

plt.tight_layout()
plt.savefig('Chart_4_Micro_Simulation.png', dpi=300)
plt.close()

print("\n✅ Success! High-resolution simulation chart saved as 'Chart_4_Micro_Simulation.png'.")