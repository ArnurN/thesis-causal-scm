import pandas as pd

print("Step 1: Loading the Minimum Viable Thesis (MVT) data...")
# Load our tiny filtered files and the calendar
sales = pd.read_csv('mvt_sales.csv')
prices = pd.read_csv('mvt_prices.csv')
calendar = pd.read_csv('calendar.csv')

print("Step 2: 'Melting' the sales data from Wide to Long format...")
# This takes the day columns (d_1, d_2) and flips them vertically into a single 'd' column
id_vars = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
sales_melted = pd.melt(sales, id_vars=id_vars, var_name='d', value_name='units_sold')

print("Step 3: Merging the Calendar to map days to weeks...")
# We join the calendar so we know which 'wm_yr_wk' (week ID) each day belongs to
sales_with_cal = pd.merge(sales_melted, calendar[['d', 'wm_yr_wk', 'date']], on='d', how='left')

print("Step 4: Merging the Prices to complete the Master Dataset...")
# Finally, we attach the exact price of the item for that specific week and store
master_data = pd.merge(sales_with_cal, prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')

# Drop any rows where we don't have a price (e.g., before the product was launched)
master_data = master_data.dropna(subset=['sell_price'])

print("Step 5: Saving the Master Dataset...")
master_data.to_csv('mvt_master.csv', index=False)

print("✅ Success! The data is merged and ready for analysis.")
print(f"Final Master Data Shape: {master_data.shape}")
print(master_data[['date', 'item_id', 'sell_price', 'units_sold']].head())