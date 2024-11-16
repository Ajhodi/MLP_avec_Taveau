# File: preprocess.py

import os
import numpy as np
import pandas as pd
from collections import Counter
import pickle

# Sliding window length
WINDOW_LENGTH = 13
data_dir = "513_distribute"  # Dataset folder
output_file = "CB513_features.pkl"  # Output file for features and labels

def extract_features(sequence, window_size):
    """Generate sliding windows from a sequence."""
    half_window = window_size // 2
    padded_seq = ['0'] * half_window + sequence + ['0'] * half_window  # Add padding
    windows = [padded_seq[i:i+window_size] for i in range(len(sequence))]
    return windows

def one_hot_encode_window(window, residues="ARNDCQEGHILKMFPSTWYV"):
    """One-hot encode a sliding window."""
    aa_to_index = {aa: i for i, aa in enumerate(residues)}
    ohe_windows = []
    for aa in window:
        ohe = np.zeros(len(residues))
        if aa in aa_to_index:
            ohe[aa_to_index[aa]] = 1
        ohe_windows.append(ohe)
    return np.array(ohe_windows).flatten()

def DSS_translate(dssp_sequence):
    """Translate DSSP annotations to 3-class (H, E, C)."""
    mapping = {'H': 'H', 'G': 'H',  # Helix
               'E': 'E', 'B': 'E',  # Strand
               'S': 'C', 'T': 'C', 'I': 'C', 'C': 'C'}  # Coil
    return ''.join([mapping.get(letter, 'C') for letter in dssp_sequence])

def process_file(filepath, window_size):
    """Extract features and labels for a single file."""
    with open(filepath, 'r', encoding='utf-8', errors='replace') as f:
        content = f.readlines()

    # Extract sequences and DSSP annotations
    RES, DSSP = None, None
    for line in content:
        if line.startswith("RES:"):
            RES = list(line.split(":")[1].strip().replace(",", ""))
        elif line.startswith("DSSP:"):
            DSSP = DSS_translate(line.split(":")[1].strip().replace(",", ""))
    
    # Ensure equal lengths
    if RES is None or DSSP is None or len(RES) != len(DSSP):
        return None, None  # Skip invalid files

    # Generate sliding windows for features
    windows = extract_features(RES, window_size)
    X = [one_hot_encode_window(w) for w in windows]  # One-hot encode windows

    # Generate labels (DSSP in 3-class format)
    y = [0 if label == 'H' else 1 if label == 'E' else 2 for label in DSSP]

    return np.array(X), np.array(y)

def preprocess_dataset(data_dir, window_size, output_file):
    """Process all files in the dataset and save features/labels."""
    X_list, y_list = [], []
    for filename in os.listdir(data_dir):
        filepath = os.path.join(data_dir, filename)
        if os.path.isfile(filepath):
            try:
                X, y = process_file(filepath, window_size)
                if X is not None and y is not None:
                    X_list.append(X)
                    y_list.append(y)
            except Exception as e:
                print(f"Error processing file {filename}: {e}")

    # Combine all data and save to file
    X = np.vstack(X_list)
    y = np.concatenate(y_list)
    with open(output_file, "wb") as f:
        pickle.dump((X, y), f)
    print(f"Preprocessed data saved to {output_file}")

# Run preprocessing
if __name__ == "__main__":
    preprocess_dataset(data_dir, WINDOW_LENGTH, output_file)
