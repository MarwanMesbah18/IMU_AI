import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

# Ensure dependencies from previous parts are available
if 'X_train' not in locals() or 'y_train' not in locals():
    print("Warning: Dependencies from Part 6 not found. This script expects X_train, y_train, X_test, y_test to be defined.")
    # In a real notebook flow, these would be present.
    # For independent testing, we might need to mock or load them.
    pass

print("\n--- Part 7: Model Optimization & Overfitting Checks ---")

# --- 1. Grid Search for Hyperparameter Tuning ---
print("\n1. performing Grid Search to find the best hyperparameters...")

# We will optimize the Random Forest as it's usually a strong baseline.
# You can easily swap this for another model if Part 6 showed something else was better.
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4] # Adding leaf constraints helps reduce overfitting
}

rf = RandomForestClassifier(random_state=42)

# 5-Fold Stratified Cross-Validation
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, 
                           cv=5, n_jobs=-1, verbose=1, scoring='accuracy')

if 'X_train' in locals():
    grid_search.fit(X_train, y_train)

    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    best_model = grid_search.best_estimator_

    print(f"\n   Best Parameters found: {best_params}")
    print(f"   Best Cross-Validation Accuracy (Mean): {best_score:.4f}")

    # Evaluate on Test Set
    y_pred_optimized = best_model.predict(X_test)
    test_acc_optimized = accuracy_score(y_test, y_pred_optimized)
    print(f"   Test Set Accuracy (Optimized Model): {test_acc_optimized:.4f}")

    # --- 2. Overfitting Check 1: CV Score vs Test Score Gap ---
    print("\n2. Overfitting Check 1: CV vs. Test Score Gap")
    
    # Gap calculation
    gap = best_score - test_acc_optimized
    print(f"   CV Mean Score: {best_score:.4f}")
    print(f"   Test Score:    {test_acc_optimized:.4f}")
    print(f"   Gap:           {gap:.4f}")

    if gap > 0.10:
        print("   WARNING: High Variance! The model performs significantly better on training/CV data than test data.")
        print("   Action: Try increasing regularization (e.g., higher min_samples_leaf, lower max_depth) or get more data.")
    elif gap < -0.02:
        print("   Note: Test score is higher than CV score. This can happen with small datasets or lucky splits.")
    else:
        print("   SUCCESS: The gap is small (<10%). The model generalizes well.")


    # --- 3. Overfitting Check 2: Learning Curves ---
    print("\n3. Overfitting Check 2: Learning Curves Analysis")
    print("   Generating Learning Curves... (This helps visualize Bias vs Variance)")

    train_sizes, train_scores, validation_scores = learning_curve(
        estimator=best_model,
        X=X_train,
        y=y_train,
        train_sizes=np.linspace(0.1, 1.0, 5),
        cv=5,
        scoring='accuracy',
        n_jobs=-1
    )

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    validation_scores_mean = np.mean(validation_scores, axis=1)
    validation_scores_std = np.std(validation_scores, axis=1)

    plt.figure(figsize=(10, 6))
    plt.title("Learning Curves (Random Forest)")
    plt.xlabel("Training Examples")
    plt.ylabel("Accuracy Score")
    plt.ylim(0.0, 1.1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, validation_scores_mean - validation_scores_std,
                     validation_scores_mean + validation_scores_std, alpha=0.1, color="g")

    plt.plot(train_sizes, train_scores_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, validation_scores_mean, 'o-', color="g", label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()

    print("   Interpretation:")
    print("   - Grid Search finds the best parameters.")
    print("   - Gap Check ensures numeric stability.")
    print("   - Learning Curves show if adding more data helps.")
    print("     * If Training Score is high but CV Score is low (Large Gap) -> Overfitting (High Variance).")
    print("     * If Both scores are low -> Underfitting (High Bias).")
    print("     * If Both scores converge to a high number -> Good Fit.")

    # --- Save Model ---
    os.makedirs('models', exist_ok=True)
    model_path = 'models/best_model.pkl'
    joblib.dump(best_model, model_path)
    print(f"\n   Optimized model saved to: {model_path}")

else:
    print("Skipping execution because X_train is not defined.")
