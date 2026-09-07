import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

print("\n" + "="*80)
print(" 🧹 PHASE 1 & 2: DATA ENGINEERING & SAMPLING")
print("="*80)

# 1. Load the raw Kaggle datasets
print("[1] Loading raw Kaggle M5 datasets...")
try:
    calendar = pd.read_csv('calendar.csv')
    prices = pd.read_csv('sell_prices.csv')
    sales = pd.read_csv('sales_train_validation.csv')
except FileNotFoundError:
    print("❌ ERROR: Raw Kaggle files not found. Ensure 'calendar.csv', 'sell_prices.csv', and 'sales_train_validation.csv' are in the directory.")
    exit()

# 2. Stratified Sampling (Pre-Melt to save RAM)
print("[2] Executing Stratified Sampling on raw items...")
# Let's filter to just one store to keep the geographical economics consistent (e.g., California Store 1)
sales = sales[sales['store_id'] == 'CA_1']

# Sample 10000 items stratified by category (Version-Proof Method)
sample_size = 10000 // sales['dept_id'].nunique()
sampled_item_ids = []

# Loop through each department and pick random items
for dept in sales['dept_id'].unique():
    dept_items = sales[sales['dept_id'] == dept]['item_id'].unique()
    # Randomly pick up to 'sample_size' items
    chosen_items = np.random.choice(dept_items, size=min(len(dept_items), sample_size), replace=False)
    sampled_item_ids.extend(chosen_items)

# Filter the main dataframe using our selected items
sampled_sales = sales[sales['item_id'].isin(sampled_item_ids)]
print(f"✅ Extracted {len(sampled_item_ids)} items.")

# 3. Data Transformation (Melting Wide to Long)
print("[3] Melting sales data from Wide to Long format...")
# Drop the ID columns we don't need, keep the day columns (d_1, d_2...)
id_vars = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
sales_long = pd.melt(sampled_sales, 
                     id_vars=id_vars, 
                     var_name='d', 
                     value_name='units_sold')

# 4. Merging the Confounders (Calendar & Holidays)
print("[4] Merging Causal Confounders (Calendar & Holidays)...")
# Select only the confounder columns we formally declared in our DAG
calendar_subset = calendar[['d', 'date', 'wm_yr_wk', 'weekday', 'wday', 'month', 'year', 'event_name_1', 'snap_CA']]
sales_long = sales_long.merge(calendar_subset, on='d', how='left')

# 5. Merging the Treatment (Price)
print("[5] Merging Treatment Variable (Price)...")
sales_long = sales_long.merge(prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# Clean up any missing prices (items that weren't sold on those days)
sales_long = sales_long.dropna(subset=['sell_price'])

# 6. Exporting the Cleaned Dataset
print("[6] Saving to 'cleaned_m5_data.csv'...")
sales_long.to_csv('cleaned_m5_data.csv', index=False)

print("\n" + "="*80)
print(f" ✅ DATA ENGINEERING COMPLETE. Dataset size: {len(sales_long)} rows.")
print(" ✅ 'cleaned_m5_data.csv' is ready for Phase 8.")
print("="*80 + "\n")