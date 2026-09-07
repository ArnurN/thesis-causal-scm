import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

print("Booting Macro-Level Thesis Visualizations (File 9) [UPDATED]...")

# ==============================================================================
# 1. LOAD THE DATA
# ==============================================================================
file_name = 'final_thesis_results_REAL.csv'

if not os.path.exists(file_name):
    print(f"Error: '{file_name}' not found in the current directory.")
    exit()

df = pd.read_csv(file_name)

# ==============================================================================
# 2. GLOBAL VISUALIZATION SETTINGS
# ==============================================================================
sns.set_theme(style="whitegrid", context="paper", font_scale=1.2)
color_siloed = "#E74C3C"  # Red
color_joint = "#2ECC71"   # Green
color_neutral = "#3498DB" # Blue

# ==============================================================================
# 3. CHART 1: THE PROFIT BOXPLOT (H1 Validation) - SEABORN SYNTAX FIXED
# ==============================================================================
plt.figure(figsize=(8, 6))
profit_data = pd.melt(df, value_vars=['Siloed_Profit', 'Joint_Profit'], 
                      var_name='Strategy', value_name='Expected Profit ($)')

# Added hue='Strategy' and legend=False to comply with new Seaborn standards
sns.boxplot(x='Strategy', y='Expected Profit ($)', data=profit_data, 
            hue='Strategy', palette=[color_siloed, color_joint], 
            legend=False, showfliers=False)

plt.title('Distribution of Expected Profit: Siloed vs. Joint Co-Optimization', fontweight='bold')
plt.ylabel('Expected Profit ($)')
plt.xlabel('Operational Strategy')
plt.tight_layout()
plt.savefig('Chart_1_Profit_Boxplot.png', dpi=300)
plt.close()

# ==============================================================================
# 4. CHART 2: THE BULLWHIP HISTOGRAM (Safety Stock Reduction)
# ==============================================================================
plt.figure(figsize=(8, 6))
clean_ss_reduction = df['SS_Reduction_Pct'].dropna()

sns.histplot(clean_ss_reduction, bins=30, color=color_neutral, kde=True)
plt.axvline(clean_ss_reduction.mean(), color='red', linestyle='dashed', linewidth=2, 
            label=f"Mean Reduction: {clean_ss_reduction.mean():.2f}%")

plt.title('Distribution of Safety Stock Reduction (Bullwhip Mitigation)', fontweight='bold')
plt.xlabel('Safety Stock Reduction (%)')
plt.ylabel('Frequency (Number of SKUs)')
plt.legend()
plt.tight_layout()
plt.savefig('Chart_2_Bullwhip_Histogram.png', dpi=300)
plt.close()

# ==============================================================================
# 5. CHART 3: CAUSAL ELASTICITY DISTRIBUTION (Upgraded for Thesis)
# ==============================================================================
plt.figure(figsize=(8, 6))

# Filter out extreme outliers for a clean academic distribution plot
elasticity_clean = df[(df['True_Elasticity'] > -10) & (df['True_Elasticity'] < 5)]['True_Elasticity']

sns.histplot(elasticity_clean, bins=40, color="#9B59B6", kde=True)
plt.axvline(elasticity_clean.mean(), color='black', linestyle='dashed', linewidth=2,
            label=f"Mean Elasticity: {elasticity_clean.mean():.2f}")
plt.axvline(0, color='red', linestyle='-', linewidth=1, alpha=0.5, label="Zero Elasticity Boundary")

plt.title('Distribution of Extracted Causal Price Elasticity (Double ML)', fontweight='bold')
plt.xlabel('Causal Price Elasticity Coefficient')
plt.ylabel('Frequency (Number of SKUs)')
plt.legend()
plt.tight_layout()
plt.savefig('Chart_3_Elasticity_Distribution.png', dpi=300)
plt.close()

print("✅ Success! Charts 1, 2, and 3 saved flawlessly as high-resolution PNGs.")