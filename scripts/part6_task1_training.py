import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score

if 'X_full' not in locals():
    print("Warning: X_full not found. Ensure Part 5 (Feature Selection) ran.")
    # Fallback or exit
    if 'df_all_features' in locals():
        X_full = df_all_features.drop(columns=['label'])
        y_full = df_all_features['label']
    else:
        # Mock for verification script only
        X_full = pd.DataFrame(np.random.rand(100, 10))
        y_full = np.array(['mock'] * 100)

print("\n--- Task 6.1: Model Training (5 Baselines) ---")

# 1. Split Data (Stratified to keep class balance)
X_train, X_test, y_train, y_test = train_test_split(
    X_full, y_full, test_size=0.2, random_state=42, stratify=y_full
)

print(f"Training Data Shape: {X_train.shape}")
print(f"Testing Data Shape: {X_test.shape}")

# 2. Define Models
models = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(kernel='rbf', probability=True, random_state=42),
    'k-NN': KNeighborsClassifier(n_neighbors=5),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'MLP': MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
}

# 3. Train and Compare
results = {}
trained_models = {}

print("\nTraining Models...")
for name, model in models.items():
    print(f"  Training {name}...", end=" ")
    try:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results[name] = acc
        trained_models[name] = model
        print(f"Done. Accuracy: {acc:.4f}")
    except Exception as e:
        print(f"Failed: {e}")

# 4. Select Best
best_model_name = max(results, key=results.get)
best_acc = results[best_model_name]
print(f"\nBest Model: {best_model_name} ({best_acc:.4f})")

# Questions
print("\n--- Questions to Answer (Task 6.1) ---")

print("a) Which model is best?")
print(f"   {best_model_name} achieved the highest accuracy of {best_acc:.2%}.")

print("b) Why try multiple models?")
print("   No free lunch theorem: Different alignments of data manifolds suit different algorithms.")
print("   RF/GBM handle non-linear/feature-interactions well. SVM is great for margins. kNN is local.")

print("c) Why use Stratified Split?")
print("   To ensure the Test set has the same proportion of activities (Walking, Jumping, etc.) as the Training set, preventing bias.")
