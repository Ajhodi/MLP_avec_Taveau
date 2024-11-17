
import os
import numpy as np
import pandas as pd

# Constants
RESIDUES = "ARNDCQEGHILKMFPSTWYV"  # Standard amino acids
DSSP_MAPPING = {'H': 0, 'E': 1, 'C': 2}  # Secondary structure mapping (Helix, Strand, Coil)
WINDOW_SIZE = 13  # Sliding window size


def dssp_to_3class(dssp_seq):
    """
    Converts DSSP 8-class annotations into 3-class labels (H, E, C).
    """
    mapping = {'H': 'H', 'G': 'H', 'E': 'E', 'B': 'E', 'S': 'C', 'T': 'C', 'I': 'C', 'C': 'C'}
    return ''.join(mapping.get(ss, 'C') for ss in dssp_seq)


def one_hot_encode(sequence, mapping):
    """
    One-hot encodes a given sequence based on a provided mapping.
    
    Parameters:
        sequence (str): Input sequence (e.g., residues or DSSP).
        mapping (dict): Mapping of characters to indices for encoding.
        
    Returns:
        np.array: One-hot encoded matrix.
    """
    seq_len = len(sequence)
    num_classes = len(mapping)
    one_hot = np.zeros((seq_len, num_classes), dtype=np.float32)
    for i, char in enumerate(sequence):
        if char in mapping:
            one_hot[i, mapping[char]] = 1
    return one_hot


def sliding_window_sequences(residues, dssp, window_size=13):
    """
    Generate sliding window subsequences from residues and DSSP labels.
    
    Parameters:
        residues (str): Amino acid sequence.
        dssp (str): DSSP secondary structure labels corresponding to the residues.
        window_size (int): The size of the sliding window.
        
    Returns:
        list: List of tuples containing subsequences of residues and their central DSSP labels.
    """
    half_window = window_size // 2
    padded_residues = f"{'X' * half_window}{residues}{'X' * half_window}"  # Pad residues with 'X'
    padded_dssp = f"{'C' * half_window}{dssp}{'C' * half_window}"  # Pad DSSP with 'C'

    subsequences = []
    for i in range(len(residues)):
        subsequence = padded_residues[i:i + window_size]
        label = padded_dssp[i + half_window]  # Use the center DSSP label as the label for this subsequence
        subsequences.append((subsequence, label))
    
    return subsequences


def parse_cb513_file(filepath):
    """
    Parses a CB513 file to extract residue and DSSP sequences.
    
    Parameters:
        filepath (str): Path to the CB513 file.
        
    Returns:
        dict: Contains extracted residue and DSSP sequences.
    """
    residues = []
    dssp = []
    try:
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith("RES:"):
                    residues.append(line.split(":")[1].strip().replace(",", ""))
                elif line.startswith("DSSP:"):
                    dssp.append(dssp_to_3class(line.split(":")[1].strip().replace(",", "")))
        
        # Ensure valid parsing
        if residues and dssp:
            return {'RES': residues[0], 'DSSP': dssp[0]}
        else:
            return None
    except Exception as e:
        print(f"Error reading file {filepath}: {e}")
        return None


def create_sliding_window_dataset(directory, window_size=13):
    """
    Reads all CB513 files from a directory and creates a structured dataset with sliding windows.
    
    Parameters:
        directory (str): Path to the directory containing CB513 files.
        window_size (int): The size of the sliding window.
        
    Returns:
        pd.DataFrame: Dataset with sliding window subsequences and corresponding DSSP labels.
    """
    subsequences = []
    for file in os.listdir(directory):
        try:
            filepath = os.path.join(directory, file)
            file_data = parse_cb513_file(filepath)
            if file_data:
                residues = file_data['RES']
                dssp = file_data['DSSP']
                subsequences.extend(sliding_window_sequences(residues, dssp, window_size))
        except Exception as e:
            print(f"Error processing file {file}: {e}")
    
    # Create DataFrame with subsequences and labels
    expanded_df = pd.DataFrame(subsequences, columns=['Subsequence', 'DSSP'])
    return expanded_df


if __name__ == "__main__":
    # Directory containing CB513 files
    DATA_DIR = "513_distribute"  # Update with your dataset directory
    
    # Generate dataset
    try:
        dataset = create_sliding_window_dataset(DATA_DIR, window_size=WINDOW_SIZE)
        print("Dataset created successfully!")
        
        # Map residues and DSSP to numerical format
        residue_mapping = {res: idx for idx, res in enumerate(RESIDUES + "X")}  # Add 'X' for padding
        dssp_mapping = DSSP_MAPPING
        
        # Convert subsequences to one-hot encoded format
        dataset['Features'] = dataset['Subsequence'].apply(
            lambda seq: one_hot_encode(seq, residue_mapping).flatten()
        )
        dataset['DSSP'] = dataset['DSSP'].map(dssp_mapping)
        
        # Save dataset to CSV
        dataset.to_csv("cb513_sliding_window_dataset.csv", index=False)
        print("Dataset saved as 'cb513_sliding_window_dataset.csv'.")
    except Exception as e:
        print(f"Error: {e}")

