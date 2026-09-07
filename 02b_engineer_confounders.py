import pandas as pd
import numpy as np
import warnings

# Suppress minor warnings for clean terminal output
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print(" ⚙️ PHASE 2B: CAUSAL FEATURE ENGINEERING (CONFOUNDERS)")
print("="*60)

# 1. Load the Master Dataset
print("\n[1] Loading the base master dataset...")
df = pd.read_csv('mvt_master.csv')
df['date'] = pd.to_datetime(df['date'])

# CRUCIAL: We must sort by item and date before calculating rolling time-series math
df = df.sort_values(by=['item_id', 'date']).reset_index(drop=True)

# 2. Engineering Confounder 1: Forward Buying (Discount Depth)
print("[2] Calculating 'discount_depth_percentage' (Forward Buying)...")
# We calculate the highest price an item was sold for over the last 30 days. This is the "Baseline Price"
df['rolling_max_price'] = df.groupby('item_id')['sell_price'].transform(lambda x: x.rolling(window=30, min_periods=1).max())

# The discount depth is the percentage difference between the baseline price and today's price
df['discount_depth_percentage'] = ((df['rolling_max_price'] - df['sell_price']) / df['rolling_max_price'])
df['discount_depth_percentage'] = df['discount_depth_percentage'].fillna(0)

# 3. Engineering Confounder 2: Demand Volatility (Bullwhip Trigger)
print("[3] Calculating 'short_term_volatility_index' (Demand Instability)...")
# We calculate the standard deviation (volatility) of the units sold over the last 7 days
df['short_term_volatility_index'] = df.groupby('item_id')['units_sold'].transform(lambda x: x.rolling(window=7, min_periods=1).std())
df['short_term_volatility_index'] = df['short_term_volatility_index'].fillna(0)

# Clean up temporary columns used for math
df = df.drop(columns=['rolling_max_price'])

# 4. Save the Enhanced Dataset
print("\n[4] Saving the enhanced master dataset...")
df.to_csv('mvt_master.csv', index=False)

# Let's print a quick preview of our new structural variables
print("\n✅ Success! Engineered Confounders Added.")
print("-" * 60)
print("Preview of the New Causal Variables:")
print(df[['date', 'item_id', 'sell_price', 'discount_depth_percentage', 'short_term_volatility_index']].tail())
print("="*60 + "\n")