import json
import os

notebook_path = 'IMU_Project.ipynb'

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

def split_and_add(nb, filename, header_map):
    if not os.path.exists(filename):
        print(f"Skipping {filename} (not found)")
        return

    with open(filename, 'r') as f:
        content = f.read()

    # defined_markers is a list of tuples: (marker_string, markdown_title_for_block)
    # logic: Find marker, split. The part *before* the marker goes to proper cell. The part *starting* at marker begins next buffer.
    
    buffer = content
    cells_to_add = []
    
    # We will process markers in order.
    # header_map = [ (marker, markdown_title), ... ]
    
    # Initial part (before first marker)
    first_marker = header_map[0][0] if header_map else None
    
    if first_marker and first_marker in buffer:
        pre_chunk, buffer = buffer.split(first_marker, 1)
        buffer = first_marker + buffer # Add marker back to start of buffer
        if pre_chunk.strip():
            cells_to_add.append(create_code_cell(pre_chunk.strip()))
    
    for i, (marker, title) in enumerate(header_map):
        # We are at the start of a block defined by 'marker'.
        # We want to find the END of this block, which is the START of the Next marker.
        next_marker = header_map[i+1][0] if i + 1 < len(header_map) else None
        
        if next_marker and next_marker in buffer:
            chunk, buffer = buffer.split(next_marker, 1)
            buffer = next_marker + buffer # Prep for next
        else:
            # Last chunk
            chunk = buffer
            buffer = ""
            
        if title:
            cells_to_add.append(create_markdown_cell(title))
        cells_to_add.append(create_code_cell(chunk.strip()))
        
        if not buffer:
            break
            
    for cell in cells_to_add:
        nb['cells'].append(cell)
    print(f"Processed {filename}: Added {len(cells_to_add)} cells.")

# Load Notebook
try:
    with open(notebook_path, 'r') as f:
        nb = json.load(f)
    print(f"Loaded notebook with {len(nb['cells'])} cells.")
except Exception as e:
    print(f"Error loading notebook: {e}")
    exit(1)

# Task 2.1
nb['cells'].append(create_markdown_cell("## Task 2.1: Handling Missing Values"))
split_and_add(nb, 'part2_task1_missing_values.py', [
    ("# Strategy A: Forward Fill", "### Strategy A: Forward Fill"),
    ("# Strategy B: Linear Interpolation", "### Strategy B: Linear Interpolation"),
    ("# Strategy C: Drop Rows", "### Strategy C: Drop Rows"),
    ("# Questions to Answer", "### Analysis & Questions")
])

# Task 2.2
nb['cells'].append(create_markdown_cell("## Task 2.2: Outlier Detection and Treatment"))
split_and_add(nb, 'part2_task2_outliers.py', [
    ("# 1. Z-Score Method", "### 1. Z-Score Method"),
    ("# 2. IQR Method", "### 2. IQR Method"),
    ("# 3. Visualization", "### 3. Visualization"),
    ("# 4. Treatment Strategies Comparison", "### 4. Treatment Strategies Comparison"),
    ("# Questions", "### Analysis & Questions")
])

# Task 2.3
nb['cells'].append(create_markdown_cell("## Task 2.3: Duplicate Removal"))
split_and_add(nb, 'part2_task3_duplicates.py', [
    ("# 1. Measurement Duplicates", "### 1. Measurement Duplicates"),
    ("# Questions", "### Analysis & Questions")
])

# Task 2.4
nb['cells'].append(create_markdown_cell("## Task 2.4: Timestamp Sorting and Resampling"))
split_and_add(nb, 'part2_task4_resampling.py', [
    ("# 1. Sort by timestamp", "### 1. Sorting & Indexing"),
    ("# 3. Resample", "### 2. Resampling (50Hz)"),
    ("# Questions", "### Analysis & Questions")
])

# Save
try:
    with open(notebook_path, 'w') as f:
        json.dump(nb, f, indent=4)
    print("Notebook updated successfully with organized cells.")
except Exception as e:
    print(f"Error saving notebook: {e}")
