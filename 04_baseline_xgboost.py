import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV, KFold
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import warnings

# Suppress minor warnings for a clean terminal output
warnings.filterwarnings('ignore')

print("\n" + "="*60)
print(" 🤖 PHASE 4: FAIR ML BASELINE (XGBOOST + CONFOUNDERS)")
print("="*60)

# 1. Load the Data
print("\n[1] Loading Enhanced Master Dataset...")
df = pd.read_csv('mvt_master.csv')

# 2. Feature Engineering
print("[2] Processing Time and Supply Chain Features...")
df['date'] = pd.to_datetime(df['date'])
df['month'] = df['date'].dt.month
df['day_of_week'] = df['date'].dt.dayofweek

columns_to_drop = ['date', 'd', 'id', 'dept_id', 'cat_id', 'store_id', 'state_id']
df = df.drop(columns=columns_to_drop)

df = pd.get_dummies(df, columns=['item_id'], drop_first=True)

X = df.drop(columns=['units_sold'])
y = df['units_sold']

print(f"    -> Features visible to XGBoost: {list(X.columns)}")

# 3. Train-Test Split (80/20)
print("\n[3] Splitting data into Training and Testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Hyperparameter Tuning with Cross-Validation
print("[4] Initializing K-Fold Cross-Validation and Grid Search...")
base_model = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)

# We keep the same grid to ensure a perfectly fair comparison to the old baseline
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.8, 1.0]
}

kf = KFold(n_splits=3, shuffle=True, random_state=42)

grid_search = GridSearchCV(
    estimator=base_model,
    param_grid=param_grid,
    scoring='neg_root_mean_squared_error',
    cv=kf,
    verbose=1,
    n_jobs=-1
)

print("    -> Tuning hyperparameters... (Testing 162 total models, please wait!)")
grid_search.fit(X_train, y_train)

best_model = grid_search.best_estimator_

print(f"\n🏆 Best Hyperparameters Found:")
for key, value in grid_search.best_params_.items():
    print(f"    - {key}: {value}")

# 5. Evaluate the Optimized Model
print("\n[5] Evaluating Optimized Model Performance...")
predictions = best_model.predict(X_test)
optimized_rmse = np.sqrt(mean_squared_error(y_test, predictions))
print(f"Optimized Root Mean Squared Error (RMSE): {optimized_rmse:.2f} units")

# 6. Extract Feature Importance
print("\n[6] Extracting Feature Importance (How the model made decisions)...")
importance = best_model.feature_importances_
feature_names = X.columns

feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importance})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

print("\nTop Features for Predicting Sales (Including Confounders):")
print(feature_importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='crimson')
plt.gca().invert_yaxis()
plt.title('Fair XGBoost Feature Importance: Correlation vs. Causation', fontweight='bold')
plt.xlabel('Relative Importance Score')
plt.tight_layout()
plt.savefig('mvt_xgboost_fair_importance.png', dpi=300)

print("\n✅ Success! Fair baseline model trained and saved as 'mvt_xgboost_fair_importance.png'.")
print("="*60 + "\n")