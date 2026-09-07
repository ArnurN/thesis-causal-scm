import pandas as pd
import os

print("Step 1: Loading datasets (this might take a minute)...")
# Load the raw data
sales = pd.read_csv('sales_train_validation.csv')
prices = pd.read_csv('sell_prices.csv')
calendar = pd.read_csv('calendar.csv')

print("Step 2: Filtering down to the Minimum Viable Thesis (MVT)...")
# We select three items that belong to the same category in California Store 1
target_store = 'CA_1'
target_items = ['FOODS_3_090', 'FOODS_3_120', 'FOODS_3_200']

# Filter sales
filtered_sales = sales[(sales['store_id'] == target_store) & (sales['item_id'].isin(target_items))]

# Filter prices
filtered_prices = prices[(prices['store_id'] == target_store) & (prices['item_id'].isin(target_items))]

print("Step 3: Saving the filtered data...")
# Save these much smaller files so we don't have to load the giant ones again!
filtered_sales.to_csv('mvt_sales.csv', index=False)
filtered_prices.to_csv('mvt_prices.csv', index=False)

print("✅ Success! Filtered Data Saved.")
print("Filtered Sales Shape:", filtered_sales.shape)
print("Filtered Prices Shape:", filtered_prices.shape)