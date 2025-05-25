"""Feature extraction and sequence processing utilities."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Union


class SequenceProcessor:
    """Process protein sequences for phosphorylation prediction."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
    
    def extract_window(self, sequence: str, position: int) -> str:
        """
        Extract a window of amino acids around a position.
        
        Args:
            sequence: Protein sequence
            position: Position to center the window on (1-based)
            
        Returns:
            Window sequence
        """
        pos_idx = position - 1  # Convert to 0-based index
        
        start = max(0, pos_idx - self.window_size)
        end = min(len(sequence), pos_idx + self.window_size + 1)
        
        window = sequence[start:end]
        return window
    
    def tokenize_sequence(self, sequence: str, tokenizer) -> Dict:
        """
        Tokenize sequence using a provided tokenizer.
        
        Args:
            sequence: Protein sequence
            tokenizer: Tokenizer to use
            
        Returns:
            Tokenized sequence
        """
        return tokenizer(
            sequence,
            padding="max_length",
            truncation=True,
            max_length=64,
            return_tensors="pt"
        )
    
    def pad_sequence(self, sequence: str, max_length: int) -> str:
        """
        Pad sequence to maximum length.
        
        Args:
            sequence: Input sequence
            max_length: Maximum length
            
        Returns:
            Padded sequence
        """
        if len(sequence) >= max_length:
            return sequence[:max_length]
        
        padding_needed = max_length - len(sequence)
        return sequence + "X" * padding_needed


class FeatureExtractor:
    """Extract various features for phosphorylation prediction."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.amino_acids = ['A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L', 
                           'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y']
        self._load_physicochemical_properties()
    
    def _load_physicochemical_properties(self):
        """Load physicochemical properties."""
        # Default properties if file not available
        self.properties = {
            'A': [1.8, 0.62, 0.046, 0.0, 0.0, 0.0, 0.0, 0.0, 0.35, 0.711, 1.18, 0.0, 0.0, 0.0, 0.0, 0.0],
            'C': [2.5, 0.29, 1.436, 0.0, 0.0, 0.0, 0.0, 0.0, 0.132, 1.069, 1.38, 0.0, 0.0, 0.0, 0.0, 0.0],
            'D': [-3.5, -0.9, 0.121, 0.0, 0.0, 0.0, 0.0, 0.0, 0.151, 0.713, 1.14, 0.0, 0.0, 0.0, 0.0, 0.0],
            'E': [-3.5, -0.74, 0.058, 0.0, 0.0, 0.0, 0.0, 0.0, 0.151, 0.637, 1.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            'F': [2.8, 1.19, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.113, 0.795, 1.20, 0.0, 0.0, 0.0, 0.0, 0.0],
            'G': [-0.4, 0.48, 0.179, 0.0, 0.0, 0.0, 0.0, 0.0, 0.881, 0.714, 1.06, 0.0, 0.0, 0.0, 0.0, 0.0],
            'H': [-3.2, -0.4, 0.23, 0.0, 0.0, 0.0, 0.0, 0.0, 0.384, 0.651, 1.22, 0.0, 0.0, 0.0, 0.0, 0.0],
            'I': [4.5, 1.38, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.131, 1.102, 1.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            'K': [-3.9, -1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.466, 0.656, 1.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            'L': [3.8, 1.06, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.131, 0.985, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            'M': [1.9, 0.64, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.228, 0.893, 1.77, 0.0, 0.0, 0.0, 0.0, 0.0],
            'N': [-3.5, -0.78, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.463, 0.663, 1.15, 0.0, 0.0, 0.0, 0.0, 0.0],
            'P': [-1.6, 0.12, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.131, 0.711, 1.22, 0.0, 0.0, 0.0, 0.0, 0.0],
            'Q': [-3.5, -0.85, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.464, 0.668, 1.17, 0.0, 0.0, 0.0, 0.0, 0.0],
            'R': [-4.5, -2.53, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.596, 0.666, 1.21, 0.0, 0.0, 0.0, 0.0, 0.0],
            'S': [-0.8, -0.18, 0.35, 0.0, 0.0, 0.0, 0.0, 0.0, 0.463, 0.669, 1.42, 0.0, 0.0, 0.0, 0.0, 0.0],
            'T': [-0.7, -0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.463, 0.674, 1.36, 0.0, 0.0, 0.0, 0.0, 0.0],
            'V': [4.2, 1.08, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.131, 0.892, 1.25, 0.0, 0.0, 0.0, 0.0, 0.0],
            'W': [-0.9, 0.81, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.085, 0.808, 1.14, 0.0, 0.0, 0.0, 0.0, 0.0],
            'Y': [-1.3, 0.26, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.204, 0.778, 1.39, 0.0, 0.0, 0.0, 0.0, 0.0],
            'X': [0.0] * 16  # Unknown amino acid
        }
    
    def extract_aac(self, sequence: str) -> Dict[str, float]:
        """Extract Amino Acid Composition (AAC) features."""
        aac = {aa: 0 for aa in self.amino_acids}
        
        seq_length = len(sequence)
        for aa in sequence:
            if aa in aac:
                aac[aa] += 1
        
        # Convert counts to frequencies
        for aa in aac:
            aac[aa] = aac[aa] / seq_length if seq_length > 0 else 0
            
        return aac
    
    def extract_dpc(self, sequence: str) -> Dict[str, float]:
        """Extract Dipeptide Composition (DPC) features."""
        dpc = {}
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                dpc[aa1 + aa2] = 0
        
        if len(sequence) < 2:
            return dpc
        
        for i in range(len(sequence) - 1):
            dipeptide = sequence[i:i+2]
            if dipeptide in dpc:
                dpc[dipeptide] += 1
        
        # Convert counts to frequencies
        total_dipeptides = len(sequence) - 1
        for dipeptide in dpc:
            dpc[dipeptide] = dpc[dipeptide] / total_dipeptides if total_dipeptides > 0 else 0
            
        return dpc
    
    def extract_tpc(self, sequence: str) -> Dict[str, float]:
        """Extract Tripeptide Composition (TPC) features."""
        tpc = {}
        
        if len(sequence) < 3:
            # Return empty tripeptide composition
            for aa1 in self.amino_acids:
                for aa2 in self.amino_acids:
                    for aa3 in self.amino_acids:
                        tpc[aa1 + aa2 + aa3] = 0
            return tpc
        
        # Count tripeptides
        for i in range(len(sequence) - 2):
            tripeptide = sequence[i:i+3]
            if all(aa in self.amino_acids for aa in tripeptide):
                if tripeptide not in tpc:
                    tpc[tripeptide] = 0
                tpc[tripeptide] += 1
        
        # Convert counts to frequencies
        total_tripeptides = len(sequence) - 2
        for tripeptide in tpc:
            tpc[tripeptide] = tpc[tripeptide] / total_tripeptides if total_tripeptides > 0 else 0
        
        # Ensure all possible tripeptides are included
        standard_tpc = {}
        for aa1 in self.amino_acids:
            for aa2 in self.amino_acids:
                for aa3 in self.amino_acids:
                    tri = aa1 + aa2 + aa3
                    standard_tpc[tri] = tpc.get(tri, 0)
        
        return standard_tpc
    
    def binary_encode_amino_acid(self, aa: str) -> List[float]:
        """Binary encode a single amino acid into a 20-dimensional vector."""
        encoding = [0.0] * 20
        
        if aa in self.amino_acids:
            idx = self.amino_acids.index(aa)
            encoding[idx] = 1.0
        
        return encoding
    
    def extract_binary_encoding(self, sequence: str, position: int) -> Dict[str, float]:
        """Extract binary encoding features for a window around the phosphorylation site."""
        pos_idx = position - 1
        
        # Define window boundaries
        start = max(0, pos_idx - self.window_size)
        end = min(len(sequence), pos_idx + self.window_size + 1)
        
        # Extract window sequence
        window = sequence[start:end]
        
        # Pad with 'X' if necessary to reach the desired window length
        left_pad = "X" * max(0, self.window_size - (pos_idx - start))
        right_pad = "X" * max(0, self.window_size - (end - pos_idx - 1))
        padded_window = left_pad + window + right_pad
        
        # Binary encode each amino acid in the window
        binary_features = {}
        for i, aa in enumerate(padded_window):
            encoding = self.binary_encode_amino_acid(aa)
            for j, value in enumerate(encoding):
                binary_features[f"BE_{i*20 + j + 1}"] = value
        
        return binary_features
    
    def extract_physicochemical_features(self, sequence: str, position: int) -> Dict[str, float]:
        """Extract physicochemical features for a window around the phosphorylation site."""
        pos_idx = position - 1
        
        # Define window boundaries
        start = max(0, pos_idx - self.window_size)
        end = min(len(sequence), pos_idx + self.window_size + 1)
        
        # Extract window sequence
        window = sequence[start:end]
        
        # Pad with 'X' if necessary to reach the desired window length
        left_pad = "X" * max(0, self.window_size - (pos_idx - start))
        right_pad = "X" * max(0, self.window_size - (end - pos_idx - 1))
        padded_window = left_pad + window + right_pad
        
        # Get properties for each amino acid in the window
        physico_features = {}
        for i, aa in enumerate(padded_window):
            if aa in self.properties:
                for j, value in enumerate(self.properties[aa]):
                    physico_features[f"PC_{i*len(self.properties[aa]) + j + 1}"] = value
            else:
                # Default values for unknown amino acids
                prop_len = len(self.properties['A'])
                for j in range(prop_len):
                    physico_features[f"PC_{i*prop_len + j + 1}"] = 0.0
        
        return physico_features
    
    def extract_all_features(self, sequence: str, position: int, config: Dict) -> np.ndarray:
        """
        Extract all features for a given sequence and position.
        
        Args:
            sequence: Protein sequence
            position: Position in sequence (1-based)
            config: Configuration dictionary specifying which features to extract
            
        Returns:
            Feature vector as numpy array
        """
        features = {}
        
        # Extract window
        window = sequence[max(0, position - 1 - self.window_size):min(len(sequence), position + self.window_size)]
        
        # Extract features based on configuration
        if config.get('use_aac', True):
            features.update(self.extract_aac(window))
        
        if config.get('use_dpc', True):
            features.update(self.extract_dpc(window))
        
        if config.get('use_tpc', True):
            features.update(self.extract_tpc(window))
        
        if config.get('use_binary', True):
            features.update(self.extract_binary_encoding(sequence, position))
        
        if config.get('use_physicochemical', True):
            features.update(self.extract_physicochemical_features(sequence, position))
        
        # Convert to sorted numpy array
        feature_names = sorted(features.keys())
        feature_values = [features[name] for name in feature_names]
        
        return np.array(feature_values)