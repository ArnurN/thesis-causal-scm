import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import warnings

warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" 🏔️ PHASE 6D: GENERATING PROFIT TOPOLOGY MAP (VISUAL PROOF)")
print("="*80)

# 1. Physics for FOODS_3_120 (From Phase 6C)
current_price = 2.50
baseline_demand = 200
sigma = 30
elasticity = -37.24

# The exact costs determined by the Newsvendor physics
c = current_price * 0.75  # Procurement
h = c * 0.30              # Holding 
b = current_price * 1.50  # Stockout

# The exact peak found by SLSQP in Phase 6C
optimal_price = 3.75
optimal_base_stock = 168

# 2. Create the Grid (The Map Boundaries)
# We test 10,000 different combinations of Price (from $2.00 to $4.50) and Inventory (from 50 to 250 units)
price_range = np.linspace(2.0, 4.5, 100)
inventory_range = np.linspace(50, 250, 100)
P_grid, S_grid = np.meshgrid(price_range, inventory_range)

# 3. Calculate Profit for all 10,000 points using the exact 6C Pure OR Math
# Calculate Demand (mu)
mu_grid = baseline_demand + (elasticity * (P_grid - current_price))
mu_grid = np.maximum(1.0, mu_grid) # Prevent negative demand

# Calculate Newsvendor Loss Function
Z_grid = (S_grid - mu_grid) / sigma
expected_shortage = sigma * (norm.pdf(Z_grid) - Z_grid * (1 - norm.cdf(Z_grid)))
expected_sales = mu_grid - expected_shortage
expected_leftovers = S_grid - expected_sales

# The Objective Function (Profit)
Profit_grid = (P_grid * expected_sales) - (c * S_grid) - (h * expected_leftovers) - (b * expected_shortage)

# 4. Draw the Contour Map
plt.figure(figsize=(12, 8))
# Draw the topological heat map
contour = plt.contourf(P_grid, S_grid, Profit_grid, levels=50, cmap='viridis')
plt.colorbar(contour, label='Expected Daily Profit ($)')

# Plot the SLSQP Peak (The Red Star)
plt.scatter(optimal_price, optimal_base_stock, color='red', marker='*', s=300, edgecolor='black', zorder=5, label=f'SLSQP Global Maximum\n(Price: ${optimal_price}, Inv: {optimal_base_stock})')

# Plot the Historical Walmart Baseline (The White Dot)
plt.scatter(current_price, baseline_demand, color='white', marker='o', s=100, edgecolor='black', zorder=5, label=f'Historical Baseline\n(Price: ${current_price}, Inv: {baseline_demand})')

# Formatting for Academic Rigor
plt.title("Visual Proof of Concavity: Joint Price & Inventory Profit Surface\n(Item: FOODS_3_120)", fontsize=14, fontweight='bold')
plt.xlabel("Retail Price ($)", fontsize=12)
plt.ylabel("Base Stock / Inventory Level (Units)", fontsize=12)
plt.axvline(optimal_price, color='red', linestyle='--', alpha=0.5)
plt.axhline(optimal_base_stock, color='red', linestyle='--', alpha=0.5)
plt.legend(loc='lower left', fontsize=10)
plt.grid(True, linestyle=':', alpha=0.6)

# Save and Show
plt.tight_layout()
plt.savefig("profit_surface_proof.png", dpi=300)
print("✅ Visual Proof saved as 'profit_surface_proof.png'.")
plt.show()

print("\n" + "="*80)