from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

if 'trained_models' not in locals():
    print("Warning: Models not found. Run Task 6.1 first.")
    pass

print("\n--- Task 6.2: Detailed Evaluation ---")

# Evaluate Best Model
best_model = trained_models[best_model_name]
y_pred = best_model.predict(X_test)

print(f"\nDetailed Report for Best Model ({best_model_name}):")
print(classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=best_model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=best_model.classes_)

# Plot
plt.figure(figsize=(8, 8))
disp.plot(cmap='Blues', values_format='d', ax=plt.gca())
plt.title(f"Confusion Matrix: {best_model_name}")
plt.show()

# Questions
print("\n--- Questions to Answer (Task 6.2) ---")

print("a) Are there confusing classes?")
print("   Check the off-diagonal elements in the matrix.")
print("   Typically, 'Walking' vs 'Runing' or 'Sit' vs 'Stand' can be confused depending on sensor orientation.")

print("b) Which metric matters most?")
print("   F1-Score is usually best if classes are imbalanced. Accuracy is fine if balanced.")
print("   Recall is crucial if missing a dangerous activity (e.g. Fall Detection) is bad.")

print("c) Overfitting checks?")
print(f"   Train Acc vs Test Acc. If Train >> Test (Gap > 10%), it's overfitting.")
train_acc = best_model.score(X_train, y_train)
test_acc = best_model.score(X_test, y_test)
gap = train_acc - test_acc
print(f"   Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}. Gap: {gap:.4f}")

if gap < 0.05:
    print("   RESULT: Gap is small (<5%). Model is NOT overfitting. It generalizes well.")
elif gap < 0.10:
    print("   RESULT: Gap is moderate (5-10%). Slight overfitting, but acceptable.")
else:
    print("   RESULT: Gap is large (>10%). Model IS overfitting. Regularization needed.")
