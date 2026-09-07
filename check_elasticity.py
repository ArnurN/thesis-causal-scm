import pandas as pd

df = pd.read_csv('final_thesis_results_REAL.csv')
print("\n🔍 VERIFYING TRUE CAUSAL ELASTICITIES")
print(df[['Item_ID', 'True_Elasticity', 'Profit_Delta_Pct']].head(10))