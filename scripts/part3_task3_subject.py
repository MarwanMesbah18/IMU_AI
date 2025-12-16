import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

if 'df' not in locals():
    # Assume loaded
    pass

print("\n--- Task 3.3: Subject Variability Analysis ---")

if 'label' not in df.columns or 'subject' not in df.columns:
    print("Missing 'label' or 'subject' columns.")
else:
    # 1. Mean Acceleration Magnitude per Subject and Activity
    # Recall we calculated 'acc_mag' in Task 3.1. If not, recalculate.
    if 'acc_mag' not in df.columns:
        df['acc_mag'] = np.sqrt(df['acc_x']**2 + df['acc_y']**2 + df['acc_z']**2)
        
    subject_stats = df.groupby(['subject', 'label'])['acc_mag'].mean().reset_index()
    print("Mean Acceleration Magnitude per Subject/Activity:")
    print(subject_stats)
    
    # 2. Grouped Bar Chart
    plt.figure(figsize=(10, 6))
    sns.barplot(data=subject_stats, x='label', y='acc_mag', hue='subject')
    plt.title("Subject Variability: Mean Acceleration Magnitude")
    plt.ylabel("Mean Magnitude (m/s^2)")
    plt.grid(True, axis='y', alpha=0.3)
    plt.show()
    
    # 3. Coefficient of Variation (CV = std/mean) for each subject
    cv_stats = df.groupby('subject')['acc_mag'].agg(['mean', 'std'])
    cv_stats['CV'] = cv_stats['std'] / cv_stats['mean']
    print("\nCoefficient of Variation (CV) per Subject:")
    print(cv_stats)

# Questions
print("\n--- Questions to Answer (Task 3.3) ---")
print("a) Do different subjects show significantly different signal amplitudes?")
print("   (Check the bar chart. Usually yes, due to sensor placement or vigor of movement).")

print("b) Should we normalize per-subject before training?")
print("   Yes. Z-score normalization per subject helps model generalization by removing user-specific intensity bias.")

print("c) How would you split this data for train/test?")
print("   Split by SUBJECT (e.g., Train on Subjects 1-3, Test on Subject 4).")
print("   Random split causes 'data leakage' where samples from the same user appear in both sets, inflating accuracy.")
