import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import mutual_info_classif, f_classif

if 'df_all_features' not in locals():
    # Assume flow
    pass

print("\n--- Task 5.3: Feature Selection and Analysis ---")

# Data
# Drop label for analysis
X_full = df_all_features.drop(columns=['label'])
y_full = df_all_features['label']
feature_names = X_full.columns.tolist()

# 1. Calculate Correlation Matrix
corr_matrix = X_full.corr().abs()

# Plot correlation matrix (heatmap subset?)
plt.figure(figsize=(12, 10))
sns.heatmap(corr_matrix, cmap='coolwarm', vmin=0, vmax=1)
plt.title("Feature Correlation Matrix")
plt.show()

# 2. Identify highly correlated pairs (> 0.9)
upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
to_drop = [column for column in upper.columns if any(upper[column] > 0.9)]

print(f"Number of highly correlated features (>0.9): {len(to_drop)}")
print("Examples of correlated features:", to_drop[:10])

# 3. Compute Feature Importance (ANOVA F-score for speed/simplicity)
# Mutual info is better but slower. Prompt says "e.g., using mutual information or ANOVA F-score"
# Let's use F-score as it's standard for classification importance ranking.
f_scores, p_values = f_classif(X_full, y_full)
# Fill NaNs
f_scores = np.nan_to_num(f_scores)

# Rank
indices = np.argsort(f_scores)[::-1]
top_k = 10
top_indices = indices[:top_k]

print("\nTop 10 Features by ANOVA F-score:")
for i in range(top_k):
    print(f"{i+1}. {feature_names[top_indices[i]]} (Score: {f_scores[top_indices[i]]:.2f})")

# 4. Select Top 10
# X_selected = X_full.iloc[:, top_indices]

# Questions
print("\n--- Questions to Answer (Task 5.3) ---")

print("a) How many feature pairs are highly correlated?")
print(f"   {len(to_drop)} features have >0.9 correlation with another feature (redundancy).")

print("b) Which features are most informative?")
print(f"   Top feature: {feature_names[top_indices[0]]}.")
print("   Typically, Magnitude Mean/Std or Frequency components at activity logic (0-2Hz for walk) are high.")

print("c) Choose only 5 features?")
print("   I'd pick unrelated high-performers to maximize info gain.")
print(f"   Likely: {feature_names[top_indices[0]]}, {feature_names[top_indices[1]]} (if not correlated), SMA, Dom Freq.")

print("d) Use PCA?")
print("   Yes. High correlation (redundancy) suggests data lives on lower manifold.")
print("   PCA would reduce dimensionality while keeping variance, simplifying the model.")
