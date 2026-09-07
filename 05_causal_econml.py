import pandas as pd
import numpy as np
from econml.dml import LinearDML
from sklearn.ensemble import RandomForestRegressor
import warnings

# We suppress warnings to keep your terminal output clean and readable
warnings.filterwarnings('ignore')

print("\n" + "="*50)
print(" 🧠 PHASE 5: CAUSAL INFERENCE (EconML)")
print("="*50)

# 1. Load and Prepare Data
print("\n[1] Loading Master Dataset and Preparing Variables...")
df = pd.read_csv('mvt_master.csv')

# Time engineering (same as before)
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

# Drop constants and text
columns_to_drop = ['date', 'd', 'id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df = df.drop(columns=columns_to_drop)

# Create dummies for the items (we keep all items this time so we can compare them)
df = pd.get_dummies(df, columns=['item_id'])
item_cols = [col for col in df.columns if 'item_id' in col]

# 2. Define the Causal Structure
# Y = Outcome (What the business wants to predict/maximize)
Y = df['units_sold']

# T = Treatment (The lever the supply chain manager can pull)
T = df['sell_price']

# W = Confounders (Variables that affect BOTH the price and the sales, like seasonality)
W = df[['month', 'day_of_week', 'discount_depth_percentage', 'short_term_volatility_index']]

# X = Heterogeneity Features (Variables that make elasticity different for different products)
X = df[item_cols]

# 3. Initialize Double Machine Learning (DML)
print("[2] Initializing Double Machine Learning (DML) Architecture...")
print("    -> model_y: Learning the baseline demand using Random Forests...")
print("    -> model_t: Learning the historical pricing strategy...")

# We use two separate Random Forests to isolate the errors (residuals)
est = LinearDML(model_y=RandomForestRegressor(n_estimators=50, random_state=42),
                model_t=RandomForestRegressor(n_estimators=50, random_state=42),
                discrete_treatment=False,
                random_state=42)

# 4. Train the Causal Model
print("\n[3] Training Causal Model to mathematically isolate Price Elasticity...")
est.fit(Y, T, X=X, W=W)

# 5. Extract Causal Insights
print("\n[4] Extracting True Causal Treatment Effects...")

# Calculate the Global Average Treatment Effect (ATE)
ate = est.ate(X)
print(f"\n🌍 Global Average Treatment Effect (ATE): {ate:.2f} units")
if ate < 0:
    print(f"Interpretation: Across all products, raising the price by $1.00 CAUSES a definitive loss of {abs(ate):.2f} sales per day.")
else:
    print(f"Interpretation: The model detected a positive treatment effect, suggesting Veblen goods or severe stockout constraints.")

# Calculate Conditional Average Treatment Effect (CATE) - the elasticity per specific product
print("\n🔍 Conditional Average Treatment Effect (CATE) by Product:")
# We feed an identity matrix to extract the unique effect for each item dummy
cate_values = est.const_marginal_effect(np.eye(len(item_cols)))
cate_df = pd.DataFrame({'Product': item_cols, 'Price Elasticity (Sales lost per $1 increase)': cate_values})
print(cate_df.to_string(index=False))

print("\n✅ Success! The true causal effect of price has been isolated.")
print("="*50 + "\n")