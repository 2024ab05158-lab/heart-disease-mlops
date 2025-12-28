"""
Exploratory Data Analysis (EDA) for Heart Disease Dataset
This script performs comprehensive EDA including data cleaning, visualization
and statistical analysis.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

# -------------------------------------------------
# Load & Clean Data
# -------------------------------------------------
print("=" * 60)
print("HEART DISEASE DATASET - EXPLORATORY DATA ANALYSIS")
print("=" * 60)

data_path = project_root / "data" / "raw" / "heart.csv"
df = pd.read_csv(data_path)

print(f"\nDataset Shape: {df.shape}")
print(f"Features: {df.shape[1] - 1}")
print(f"Samples: {df.shape[0]}")

# Replace '?' with NaN
df.replace("?", np.nan, inplace=True)

# Convert to numeric
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Binary target: 1 = disease present, 0 = absent
df["target"] = df["target"].apply(lambda x: 1 if x > 0 else 0)

print("\n" + "=" * 60)
print("DATA OVERVIEW")
print("=" * 60)
print("\nFirst 5 rows:")
print(df.head())

print("\n\nDataset Info:")
print(df.info())

print("\n\nMissing Values:")
missing = df.isnull().sum()
print(missing[missing > 0] if missing.sum() > 0 else "No missing values found")

print("\n\nBasic Statistics:")
print(df.describe())

# -------------------------------------------------
# Class Distribution Analysis
# -------------------------------------------------
print("\n" + "=" * 60)
print("CLASS DISTRIBUTION")
print("=" * 60)
class_counts = df["target"].value_counts()
print(f"\nClass Distribution:")
print(f"No Disease (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
print(f"Disease (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")

# Visualize class distribution
plt.figure(figsize=(8, 5))
sns.countplot(x="target", data=df, palette="Set2")
plt.title("Class Distribution (Heart Disease Presence)", fontsize=14, fontweight='bold')
plt.xlabel("Target (0 = No Disease, 1 = Disease)", fontsize=12)
plt.ylabel("Count", fontsize=12)
plt.xticks([0, 1], ['No Disease', 'Disease'])
for i, v in enumerate(class_counts.values):
    plt.text(i, v + 5, str(v), ha='center', fontsize=11, fontweight='bold')
plt.tight_layout()
plt.savefig(project_root / "notebooks" / "class_distribution.png", dpi=300, bbox_inches='tight')
print("\nSaved: notebooks/class_distribution.png")
plt.close()

# -------------------------------------------------
# Feature Distribution Analysis
# -------------------------------------------------
print("\n" + "=" * 60)
print("FEATURE DISTRIBUTIONS")
print("=" * 60)

# Select numeric features for histogram
numeric_features = df.select_dtypes(include=[np.number]).columns.tolist()
if 'target' in numeric_features:
    numeric_features.remove('target')

# Create histograms for key features
n_features = len(numeric_features)
n_cols = 4
n_rows = (n_features + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4*n_rows))
axes = axes.flatten() if n_features > 1 else [axes]

for idx, feature in enumerate(numeric_features[:len(axes)]):
    ax = axes[idx]
    df[feature].hist(bins=30, ax=ax, alpha=0.7, color='skyblue', edgecolor='black')
    ax.set_title(f'{feature}', fontsize=11, fontweight='bold')
    ax.set_xlabel('Value', fontsize=10)
    ax.set_ylabel('Frequency', fontsize=10)
    ax.grid(True, alpha=0.3)

# Hide extra subplots
for idx in range(len(numeric_features), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Feature Distributions', fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(project_root / "notebooks" / "feature_distributions.png", dpi=300, bbox_inches='tight')
print("Saved: notebooks/feature_distributions.png")
plt.close()

# -------------------------------------------------
# Correlation Analysis
# -------------------------------------------------
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS")
print("=" * 60)

# Calculate correlation matrix
corr_matrix = df.corr()

# Correlation with target
target_corr = corr_matrix['target'].sort_values(ascending=False)
print("\nTop features correlated with target:")
print(target_corr[target_corr.index != 'target'].head(10))

# Visualize correlation heatmap
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(
    corr_matrix,
    mask=mask,
    annot=True,
    fmt='.2f',
    cmap='coolwarm',
    center=0,
    square=True,
    linewidths=1,
    cbar_kws={"shrink": 0.8},
    vmin=-1,
    vmax=1
)
plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(project_root / "notebooks" / "correlation_heatmap.png", dpi=300, bbox_inches='tight')
print("\nSaved: notebooks/correlation_heatmap.png")
plt.close()

# -------------------------------------------------
# Target vs Features Analysis
# -------------------------------------------------
print("\n" + "=" * 60)
print("TARGET VS FEATURES ANALYSIS")
print("=" * 60)

# Box plots for key features vs target
key_features = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
n_key = len(key_features)
fig, axes = plt.subplots(1, n_key, figsize=(20, 4))

for idx, feature in enumerate(key_features):
    ax = axes[idx] if n_key > 1 else axes
    df.boxplot(column=feature, by='target', ax=ax)
    ax.set_title(f'{feature} by Target', fontsize=11, fontweight='bold')
    ax.set_xlabel('Target', fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=0)

if n_key > 1:
    plt.suptitle('')
plt.suptitle('Feature Distributions by Target Class', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(project_root / "notebooks" / "target_vs_features.png", dpi=300, bbox_inches='tight')
print("Saved: notebooks/target_vs_features.png")
plt.close()

# -------------------------------------------------
# Summary Statistics by Class
# -------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY STATISTICS BY CLASS")
print("=" * 60)
print("\nNo Disease (0):")
print(df[df['target'] == 0][numeric_features].describe())
print("\n\nDisease (1):")
print(df[df['target'] == 1][numeric_features].describe())

print("\n" + "=" * 60)
print("EDA COMPLETE!")
print("=" * 60)
print("\nGenerated visualizations:")
print("- notebooks/class_distribution.png")
print("- notebooks/feature_distributions.png")
print("- notebooks/correlation_heatmap.png")
print("- notebooks/target_vs_features.png")
