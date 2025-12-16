import json
import os
import re

notebook_path = 'IMU_Project.ipynb'

# Requirement Definitions (Part 1 - 5)
requirements_text = {
    # PART 1
    "1.1": """### Task 1.1: Load and Inspect the Data
**Requirements:**
1. Load `imu_messy_data.csv` into a pandas DataFrame.
2. Display the first 20 rows.
3. Print dataset shape, column names, and data types.
4. Generate summary statistics using `describe()`.

**Questions to answer:**
a) How many samples are in the dataset?
b) What is the range of timestamp values?
c) What are the min/max values for each acceleration axis? Do they seem physically plausible?
d) What is the class distribution across activities?""",

    "1.2": """### Task 1.2: Missing Value Analysis
**Requirements:**
1. Count and report missing values per column.
2. Calculate the percentage of missing data for each accelerometer axis.
3. Identify rows where all three axes are missing simultaneously.
4. Visualize the distribution of missing values.

**Questions to answer:**
a) Which axis has the most missing values?
b) Are missing values randomly distributed or clustered?
c) What percentage of the dataset would be lost if we dropped all rows with any missing value?""",

    "1.3": """### Task 1.3: Timestamp Analysis
**Requirements:**
1. Check if timestamps are sorted in ascending order.
2. Calculate the time differences between consecutive samples.
3. Identify gaps larger than expected (assuming 50 Hz sampling rate).
4. Detect timestamp duplicates.

**Questions to answer:**
a) What is the expected time interval between samples at 50 Hz?
b) How many samples have timestamp gaps exceeding 1 second?
c) What is the largest gap in the data? Where does it occur?
d) Are there any duplicate timestamps?""",

    # PART 2
    "2.1": """### Task 2.1: Handling Missing Values
**Requirements:**
1. Implement and compare three strategies for handling missing values:
    *   **Strategy A: Forward Fill**: Use `fillna(method='ffill')`.
    *   **Strategy B: Linear Interpolation**: Use `interpolate(method='linear')`.
    *   **Strategy C: Drop Rows**: Remove all rows containing any NaN values.""",

    "2.2": """### Task 2.2: Outlier Detection and Treatment
**Requirements:**
1. Implement z-score based outlier detection (threshold: |z| > 3).
2. Implement IQR-based outlier detection.
3. Count outliers detected by each method for each axis.
4. Visualize outliers using box plots and scatter plots.

**Treatment strategies to compare:**
1. Removal
2. Clipping
3. Interpolation""",

    "2.3": """### Task 2.3: Duplicate Removal
**Requirements:**
1. Identify exact duplicate rows.
2. Identify duplicate measurement rows.
3. Remove duplicates keeping the first occurrence.
4. Document counts.""",

    "2.4": """### Task 2.4: Timestamp Sorting and Resampling
**Requirements:**
1. Sort the dataset by timestamp in ascending order.
2. Set timestamp as the DataFrame index.
3. Resample to a fixed 50 Hz rate using appropriate interpolation.
4. Handle the major gap in the data appropriately.""",

    # PART 3
    "3.1": """### Task 3.1: Time-Series Visualization
**Requirements:**
1. Plot a 10-second segment of all three axes for each activity class.
2. Create a multi-panel figure showing one representative segment per activity.
3. Color-code by activity label.
4. Plot the acceleration magnitude.""",

    "3.2": """### Task 3.2: Statistical Analysis per Class
**Requirements:**
1. Calculate mean, standard deviation, min, and max for each axis grouped by activity.
2. Create box plots for each axis grouped by activity.
3. Plot histograms of each axis for each activity.
4. Calculate correlation matrices for each activity.""",

    "3.3": """### Task 3.3: Subject Variability Analysis
**Requirements:**
1. Calculate mean acceleration magnitude for each subject and activity combination.
2. Create a grouped bar chart showing subject differences.
3. Compute coefficient of variation (CV) for each subject.""",

    "3.4": """### Task 3.4: Label Noise Investigation
**Requirements:**
1. Calculate mean and std of acceleration magnitude for each activity.
2. Identify samples whose magnitude deviates significantly from their class mean.
3. Manually inspect 10-20 suspicious samples.""",

    # PART 4
    "4.1": """### Task 4.1: Normalization Strategies
**Requirements:**
1. Implement and compare: Per-Axis Z-score, Min-Max Scaling, Per-Subject Standardization.""",

    "4.2": """### Task 4.2: Sliding Window Segmentation
**Requirements:**
1. Implement sliding window (2s, 50% overlap).
2. Assign label (majority voting).
3. Count windows per activity.
4. Experiment with window sizes (1s, 2s, 4s).""",

    "4.3": """### Task 4.3: Handling Gaps and Missing Segments
**Requirements:**
1. Identify major gap.
2. Split dataset into sessions.
3. Report durations.""",

    # PART 5
    "5.1": """### Task 5.1: Time-Domain Features
**Requirements:**
1. Extract features per window: Mean, Std, Min, Max, Range, SMA, Energy, ZCR, RMS.
2. Document total features.""",

    "5.2": """### Task 5.2: Frequency-Domain Features
**Requirements:**
1. Apply FFT.
2. Compute: Dominant freq, Spectral energy, entropy, Band powers.
3. Visualize PSD.""",

    "5.3": """### Task 5.3: Feature Selection and Analysis
**Requirements:**
1. Calculate correlation matrix.
2. Identify highly correlated (>0.9).
3. Compute feature importance (ANOVA).
4. Select top 10 features."""
}

def create_markdown_cell(source):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": [line + "\n" for line in source.split('\n')]
    }

def create_code_cell(source):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [line + "\n" for line in source.split('\n')]
    }

def process_qa_grouped(nb, qa_code):
    lines = qa_code.split('\n')
    questions = []
    
    # Regex for print("x) ...")
    q_pattern = re.compile(r'^\s*print\(f?[\'"]([a-z])\)\s*(.+)[\'"]\)\s*$')
    
    for line in lines:
        match = q_pattern.match(line)
        if match:
            letter = match.group(1)
            text = match.group(2)
            questions.append(f"**{letter})** {text}")
            
    if questions:
        md_content = "### Analysis & Questions\n\n" + "\n\n".join(questions)
        nb['cells'].append(create_markdown_cell(md_content))
    else:
        # Fallback
        nb['cells'].append(create_markdown_cell("### Analysis & Questions"))

    nb['cells'].append(create_code_cell(qa_code.strip()))

# Initialize New Notebook Structure
nb = {
    "cells": [],
    "metadata": {
        "kernelspec": {
           "display_name": "Python 3",
           "language": "python",
           "name": "python3"
        },
        "language_info": {
           "codemirror_mode": {
            "name": "ipython",
            "version": 3
           },
           "file_extension": ".py",
           "mimetype": "text/x-python",
           "name": "python",
           "nbconvert_exporter": "python",
           "pygments_lexer": "ipython3",
           "version": "3.x"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
}

# Add Title
nb['cells'].append(create_markdown_cell("# IMU Data Processing Project"))

# Standard Imports Cell
setup_code = """import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import signal, stats
from scipy.fft import fft, fftfreq
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import f_classif

# Configure plotting
# %matplotlib inline
plt.rcParams['figure.figsize'] = (10, 6)
"""
nb['cells'].append(create_code_cell(setup_code))

def split_and_add_grouped(nb, filename):
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        return

    with open(filename, 'r') as f:
        lines = f.readlines()

    # Filter out imports
    filtered_lines = []
    for line in lines:
        if line.strip().startswith(('import ', 'from ')):
            continue
        filtered_lines.append(line)
    
    content = "".join(filtered_lines)
    parts = content.split("# Questions")
    
    # 1. Implementation
    if parts[0].strip():
        nb['cells'].append(create_code_cell(parts[0].strip()))
    
    # 2. Q&A
    if len(parts) > 1:
        qa_content = parts[1]
        process_qa_grouped(nb, qa_content)
        
    print(f"Processed {filename} (Imports stripped)")

# PART 1
nb['cells'].append(create_markdown_cell("## Part 1: Data Loading and Initial Exploration"))
nb['cells'].append(create_markdown_cell(requirements_text["1.1"]))
split_and_add_grouped(nb, 'scripts/part1_task1_load.py')
nb['cells'].append(create_markdown_cell(requirements_text["1.2"]))
split_and_add_grouped(nb, 'scripts/part1_task2_missing.py')
nb['cells'].append(create_markdown_cell(requirements_text["1.3"]))
split_and_add_grouped(nb, 'scripts/part1_task3_timestamp.py')

# PART 2
nb['cells'].append(create_markdown_cell("## Part 2: Data Cleaning"))
nb['cells'].append(create_markdown_cell(requirements_text["2.1"]))
split_and_add_grouped(nb, 'scripts/part2_task1_missing_values.py')
nb['cells'].append(create_markdown_cell(requirements_text["2.2"]))
split_and_add_grouped(nb, 'scripts/part2_task2_outliers.py')
nb['cells'].append(create_markdown_cell(requirements_text["2.3"]))
split_and_add_grouped(nb, 'scripts/part2_task3_duplicates.py')
nb['cells'].append(create_markdown_cell(requirements_text["2.4"]))
split_and_add_grouped(nb, 'scripts/part2_task4_resampling.py')

# PART 3
nb['cells'].append(create_markdown_cell("## Part 3: Exploratory Data Analysis and Visualization"))
nb['cells'].append(create_markdown_cell(requirements_text["3.1"]))
split_and_add_grouped(nb, 'scripts/part3_task1_viz.py')
nb['cells'].append(create_markdown_cell(requirements_text["3.2"]))
split_and_add_grouped(nb, 'scripts/part3_task2_stats.py')
nb['cells'].append(create_markdown_cell(requirements_text["3.3"]))
split_and_add_grouped(nb, 'scripts/part3_task3_subject.py')
nb['cells'].append(create_markdown_cell(requirements_text["3.4"]))
split_and_add_grouped(nb, 'scripts/part3_task4_label_noise.py')

# PART 4
nb['cells'].append(create_markdown_cell("## Part 4: Preprocessing and Segmentation"))
nb['cells'].append(create_markdown_cell(requirements_text["4.1"]))
split_and_add_grouped(nb, 'scripts/part4_task1_normalization.py')
nb['cells'].append(create_markdown_cell(requirements_text["4.2"]))
split_and_add_grouped(nb, 'scripts/part4_task2_segmentation.py')
nb['cells'].append(create_markdown_cell(requirements_text["4.3"]))
split_and_add_grouped(nb, 'scripts/part4_task3_gaps.py')

# PART 5
nb['cells'].append(create_markdown_cell("## Part 5: Feature Engineering"))
nb['cells'].append(create_markdown_cell(requirements_text["5.1"]))
split_and_add_grouped(nb, 'scripts/part5_task1_time.py')
nb['cells'].append(create_markdown_cell(requirements_text["5.2"]))
split_and_add_grouped(nb, 'scripts/part5_task2_freq.py')
nb['cells'].append(create_markdown_cell(requirements_text["5.3"]))
split_and_add_grouped(nb, 'scripts/part5_task3_selection.py')

# Save
try:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=4)
    print("Full Notebook (Parts 1-5) Generated Successfully.")
except Exception as e:
    print(f"Error saving notebook: {e}")
