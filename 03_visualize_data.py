import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

print("\n" + "="*50)
print(" 📊 PHASE 3: EXPLORATORY DATA ANALYSIS (EDA)")
print("="*50)

# 1. Load the Data
print("\n[1] Loading Master Dataset...")
df = pd.read_csv('mvt_master.csv')

# 2. Analytical Text Description (Terminal Output)
print("\n[2] Executing Statistical Analysis...")
print("-" * 30)

# Calculate the overall correlation between price and units sold
correlation = df['sell_price'].corr(df['units_sold'])
print(f"Global Price-to-Sales Correlation: {correlation:.3f}")

if correlation < 0:
    print("💡 Analytical Insight: The correlation is negative. This confirms standard economic theory: "
          "as the price increases, consumer demand (units sold) decreases.")
else:
    print("💡 Analytical Insight: The correlation is positive/neutral, suggesting inelastic demand or "
          "confounding variables (like holidays) affecting these specific items.")

print("-" * 30)

# Let's look at one specific item to see how its price fluctuates
target_item = 'FOODS_3_090'
item_data = df[df['item_id'] == target_item]
unique_prices = item_data['sell_price'].unique()

print(f"\n[3] Deep Dive: Item {target_item}")
print(f"Over the recorded period, this product was sold at {len(unique_prices)} different price points.")
print(f"Lowest Price: ${min(unique_prices):.2f} | Highest Price: ${max(unique_prices):.2f}")

# 3. Generating the Visualizations
print("\n[4] Generating Publication-Ready Visualizations...")

# Set the visual style for academic charts
sns.set_theme(style="whitegrid")

# Create a figure with two subplots side-by-side
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Chart 1: Scatter plot with a regression line (Price Elasticity)
sns.regplot(data=df, x='sell_price', y='units_sold', 
            scatter_kws={'alpha':0.3, 'color':'blue'}, 
            line_kws={'color':'red', 'linewidth':2}, 
            ax=axes[0])
axes[0].set_title('Price vs. Units Sold (All MVT Items)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Sell Price ($)', fontsize=12)
axes[0].set_ylabel('Units Sold (Daily)', fontsize=12)

# Chart 2: Distribution of Sales Volume
sns.histplot(data=df, x='units_sold', bins=30, kde=True, color='purple', ax=axes[1])
axes[1].set_title('Distribution of Daily Sales Volume', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Units Sold', fontsize=12)
axes[1].set_ylabel('Frequency (Days)', fontsize=12)

# Save the plot to your folder
plt.tight_layout()
plt.savefig('mvt_eda_dashboard.png', dpi=300)

print("✅ Success! Visualizations saved as 'mvt_eda_dashboard.png'.")
print("="*50 + "\n")