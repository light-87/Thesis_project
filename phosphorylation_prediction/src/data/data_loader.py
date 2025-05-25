"""Data loading utilities for phosphorylation prediction."""

import os
import random
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Iterator, Optional
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold
from torch.utils.data import DataLoader
import gc


def load_sequences(file_path: str = "Sequence_data.txt") -> pd.DataFrame:
    """
    Load protein sequences from a FASTA file.
    
    Args:
        file_path: Path to FASTA file
        
    Returns:
        DataFrame with protein sequences
    """
    print("Loading protein sequences...")
    
    headers = []
    sequences = []
    current_header = None
    current_seq = ""
    
    with open(file_path, "r") as file:
        for line in file:
            line = line.strip()
            if line.startswith(">"):
                # If there is an existing sequence, save it
                if current_header:
                    headers.append(current_header)
                    sequences.append(current_seq)
                
                # Extract header ID (middle part)
                full_header = line[1:]
                parts = full_header.split("|")
                current_header = parts[1] if len(parts) > 1 else full_header
                current_seq = ""
            else:
                current_seq += line
        
        # Don't forget the last sequence
        if current_header:
            headers.append(current_header)
            sequences.append(current_seq)
    
    # Create DataFrame
    df = pd.DataFrame({
        "Header": headers,
        "Sequence": sequences
    })
    
    print(f"Loaded {len(df)} protein sequences")
    return df


def load_labels(file_path: str = "labels.xlsx") -> pd.DataFrame:
    """
    Load phosphorylation site labels.
    
    Args:
        file_path: Path to labels file
        
    Returns:
        DataFrame with phosphorylation site labels
    """
    print("Loading phosphorylation site labels...")
    
    if file_path.endswith('.xlsx'):
        df_labels = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df_labels = pd.read_csv(file_path)
    else:
        raise ValueError(f"Unsupported file format: {file_path}")
    
    print(f"Loaded {len(df_labels)} phosphorylation sites")
    return df_labels


def merge_sequence_and_labels(df_seq: pd.DataFrame, df_labels: pd.DataFrame) -> pd.DataFrame:
    """
    Merge sequence data with labels data.
    
    Args:
        df_seq: Sequence DataFrame
        df_labels: Labels DataFrame
        
    Returns:
        Merged DataFrame
    """
    print("Merging sequences with labels...")
    
    # Merge using pandas
    merged_df = pd.merge(
        df_seq,
        df_labels,
        left_on="Header",
        right_on="UniProt ID",
        how="inner"
    )
    
    # Add target column
    merged_df["target"] = 1  # All these are positive examples
    
    print(f"Merged data contains {len(merged_df)} rows")
    return merged_df


def generate_negative_samples(df_merged: pd.DataFrame, random_seed: int = 42) -> pd.DataFrame:
    """
    Generate negative samples for each protein sequence by randomly sampling
    from S/T/Y sites that are not known phosphorylation sites.
    
    Args:
        df_merged: Merged DataFrame with positive samples
        random_seed: Random seed for reproducibility
        
    Returns:
        DataFrame with both positive and negative samples
    """
    print("Generating negative samples...")
    
    # Set random seed for reproducibility
    random.seed(random_seed)
    np.random.seed(random_seed)
    
    all_rows = []
    
    # Group by Header (Protein ID)
    for header, group in df_merged.groupby('Header'):
        # Extract sequence from the first row
        seq = group['Sequence'].iloc[0]
        
        # Get positive positions from the group
        positive_positions = group['Position'].astype(int).tolist()
        
        # Find all S/T/Y positions in the sequence
        sty_positions = [i+1 for i, aa in enumerate(seq) if aa in ["S", "T", "Y"]]
        
        # Exclude the positives to get negative candidates
        negative_candidates = [pos for pos in sty_positions if pos not in positive_positions]
        
        # Number of positives for this sequence
        n_pos = len(positive_positions)
        
        # Sample negative sites (same number as positives if possible)
        sample_size = min(n_pos, len(negative_candidates))
        if sample_size > 0:
            # Use deterministic seed for this protein
            random.seed(random_seed + hash(header) % 10000)
            sampled_negatives = random.sample(negative_candidates, sample_size)
            
            # Add original positive rows to the result
            all_rows.append(group)
            
            # Create new rows for negative sites
            for neg_pos in sampled_negatives:
                # Create a copy of the first row as a template
                new_row = group.iloc[0].copy()
                
                # Update the specific fields
                new_row['AA'] = seq[neg_pos - 1]
                new_row['Position'] = neg_pos
                new_row['target'] = 0
                
                # Add the new row
                all_rows.append(pd.DataFrame([new_row]))
    
    # Combine all rows
    df_final = pd.concat(all_rows, ignore_index=True)
    
    print(f"Generated data with {len(df_final)} rows (includes both positive and negative samples)")
    return df_final


def clean_data(df_merged: pd.DataFrame, max_seq_length: int = 5000) -> pd.DataFrame:
    """
    Clean the merged data by removing sequences that are too long.
    
    Args:
        df_merged: Merged DataFrame
        max_seq_length: Maximum allowed sequence length
        
    Returns:
        Cleaned DataFrame
    """
    print(f"Cleaning data (removing sequences longer than {max_seq_length})...")
    
    # Calculate sequence lengths and filter
    df_merged['SeqLength'] = df_merged['Sequence'].str.len()
    df_cleaned = df_merged[df_merged['SeqLength'] <= max_seq_length].copy()
    df_cleaned = df_cleaned.drop('SeqLength', axis=1)
    
    print(f"After cleaning, data contains {len(df_cleaned)} rows")
    return df_cleaned


def split_dataset_by_protein(df: pd.DataFrame, 
                           train_ratio: float = 0.7, 
                           val_ratio: float = 0.15, 
                           test_ratio: float = 0.15,
                           random_seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split dataset by protein to avoid data leakage.
    
    Args:
        df: Input DataFrame
        train_ratio: Fraction for training set
        val_ratio: Fraction for validation set
        test_ratio: Fraction for test set
        random_seed: Random seed for reproducibility
        
    Returns:
        Tuple of (train_df, val_df, test_df)
    """
    print("Splitting data to avoid protein leakage...")
    
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Get all unique protein headers
    headers = df['Header'].unique()
    n_headers = len(headers)
    
    # Shuffle headers using random seed
    np.random.seed(random_seed)
    np.random.shuffle(headers)
    
    # Split points
    train_split = int(n_headers * train_ratio)
    val_split = int(n_headers * (train_ratio + val_ratio))
    
    # Split headers
    train_headers = headers[:train_split]
    val_headers = headers[train_split:val_split]
    test_headers = headers[val_split:]
    
    # Create dataframes
    train_df = df[df['Header'].isin(train_headers)].copy()
    val_df = df[df['Header'].isin(val_headers)].copy()
    test_df = df[df['Header'].isin(test_headers)].copy()
    
    print(f"Train set: {len(train_df)} samples from {len(train_headers)} proteins")
    print(f"Validation set: {len(val_df)} samples from {len(val_headers)} proteins")
    print(f"Test set: {len(test_df)} samples from {len(test_headers)} proteins")
    
    return train_df, val_df, test_df


def create_cv_splits(data: pd.DataFrame, config: Dict) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Create cross-validation splits ensuring proteins don't leak across folds.
    
    Args:
        data: Input DataFrame
        config: Configuration dictionary
        
    Returns:
        List of (train_indices, val_indices) tuples
    """
    n_folds = config.get('n_folds', 5)
    strategy = config.get('strategy', 'stratified_group')
    
    # Get protein groups and targets
    groups = data[config.get('protein_id_col', 'Header')].values
    targets = data[config.get('target_col', 'target')].values
    
    if strategy == 'stratified_group':
        # Use StratifiedGroupKFold to maintain class balance while respecting protein groups
        cv = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=42)
        splits = list(cv.split(data, targets, groups))
    elif strategy == 'group':
        # Use GroupKFold to respect protein groups only
        cv = GroupKFold(n_splits=n_folds)
        splits = list(cv.split(data, targets, groups))
    else:
        raise ValueError(f"Unsupported CV strategy: {strategy}")
    
    return splits


def create_data_loader(dataset, config: Dict, is_training: bool = True) -> DataLoader:
    """
    Create DataLoader for a dataset.
    
    Args:
        dataset: Dataset object
        config: Configuration dictionary
        is_training: Whether this is for training
        
    Returns:
        DataLoader object
    """
    batch_size = config.get('batch_size', 32)
    num_workers = config.get('num_workers', 4)
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_training,
        num_workers=num_workers,
        pin_memory=True if config.get('device', 'cpu') == 'cuda' else False,
        drop_last=is_training
    )


def load_raw_data(sequence_file: str, labels_file: str, 
                 max_seq_length: int = 5000) -> pd.DataFrame:
    """
    Load and process raw data files.
    
    Args:
        sequence_file: Path to sequence file
        labels_file: Path to labels file
        max_seq_length: Maximum sequence length
        
    Returns:
        Processed DataFrame with positive and negative samples
    """
    # Load sequences and labels
    df_seq = load_sequences(sequence_file)
    df_labels = load_labels(labels_file)
    
    # Merge data
    df_merged = merge_sequence_and_labels(df_seq, df_labels)
    
    # Clean up original dataframes
    del df_seq, df_labels
    gc.collect()
    
    # Clean data
    df_merged = clean_data(df_merged, max_seq_length)
    
    # Generate negative samples
    df_final = generate_negative_samples(df_merged)
    
    # Clean up
    del df_merged
    gc.collect()
    
    print(f"Final dataset contains {len(df_final)} samples")
    print("Class distribution:")
    print(df_final['target'].value_counts())
    
    return df_final


def stratified_group_split(X: np.ndarray, y: np.ndarray, groups: np.ndarray, 
                          n_splits: int = 5) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """
    Custom stratified group split that maintains class balance while respecting groups.
    
    Args:
        X: Feature matrix
        y: Target vector
        groups: Group labels
        n_splits: Number of splits
        
    Yields:
        Train and validation indices for each fold
    """
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    for train_idx, val_idx in cv.split(X, y, groups):
        yield train_idx, val_idx