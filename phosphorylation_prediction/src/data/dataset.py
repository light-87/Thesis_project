"""Dataset classes for phosphorylation prediction."""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from typing import Dict, List, Optional, Tuple, Union
from .preprocessor import SequenceProcessor, FeatureExtractor


class PhosphoPredictionDataset(Dataset):
    """Base dataset class for protein sequences and phosphorylation sites."""
    
    def __init__(self, data_path: str, config: Dict, tokenizer=None):
        """
        Initialize dataset.
        
        Args:
            data_path: Path to data file
            config: Configuration dictionary
            tokenizer: Optional tokenizer for transformer models
        """
        self.config = config
        self.tokenizer = tokenizer
        self.window_size = config.get('window_size', 10)
        self.max_length = config.get('max_length', 64)
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            self.data = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Initialize processors
        self.sequence_processor = SequenceProcessor(self.window_size)
        
        # Get column names from config
        self.sequence_col = config.get('sequence_col', 'Sequence')
        self.position_col = config.get('position_col', 'Position')
        self.target_col = config.get('target_col', 'target')
        self.protein_id_col = config.get('protein_id_col', 'Header')
        
        # Validate required columns exist
        required_cols = [self.sequence_col, self.position_col, self.target_col, self.protein_id_col]
        for col in required_cols:
            if col not in self.data.columns:
                raise ValueError(f"Required column '{col}' not found in data")
    
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Dict:
        row = self.data.iloc[idx]
        sequence = row[self.sequence_col]
        position = int(row[self.position_col])
        target = int(row[self.target_col])
        protein_id = row[self.protein_id_col]
        
        # Extract window around the phosphorylation site
        window_sequence = self.sequence_processor.extract_window(sequence, position)
        
        sample = {
            'sequence': sequence,
            'window_sequence': window_sequence,
            'position': position,
            'target': target,
            'protein_id': protein_id
        }
        
        # Add tokenized sequence if tokenizer provided
        if self.tokenizer is not None:
            encoding = self.tokenizer(
                window_sequence,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            sample['input_ids'] = encoding['input_ids'].squeeze(0)
            sample['attention_mask'] = encoding['attention_mask'].squeeze(0)
            sample['target'] = torch.tensor(target, dtype=torch.float)
        
        return sample
    
    def get_protein_groups(self) -> List[str]:
        """Return list of protein IDs for group-based splitting."""
        return self.data[self.protein_id_col].tolist()


class FeatureDataset(Dataset):
    """Dataset for pre-extracted features."""
    
    def __init__(self, features: Union[np.ndarray, pd.DataFrame], labels: np.ndarray, config: Dict):
        """
        Initialize feature dataset.
        
        Args:
            features: Feature matrix
            labels: Target labels
            config: Configuration dictionary
        """
        self.config = config
        
        if isinstance(features, pd.DataFrame):
            self.feature_names = features.columns.tolist()
            self.features = features.values
        else:
            self.features = features
            self.feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        self.labels = labels
        
        # Normalize features if requested
        if config.get('normalize_features', True):
            self.normalize_features()
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        label = torch.tensor(self.labels[idx], dtype=torch.float32)
        return features, label
    
    def normalize_features(self) -> None:
        """Normalize feature values to zero mean and unit variance."""
        self.feature_mean = np.mean(self.features, axis=0)
        self.feature_std = np.std(self.features, axis=0)
        
        # Avoid division by zero
        self.feature_std[self.feature_std == 0] = 1.0
        
        self.features = (self.features - self.feature_mean) / self.feature_std
    
    def get_feature_names(self) -> List[str]:
        """Return feature column names."""
        return self.feature_names


class TransformerDataset(PhosphoPredictionDataset):
    """Specialized dataset for transformer models."""
    
    def __init__(self, data_path: str, config: Dict, tokenizer):
        """
        Initialize transformer dataset.
        
        Args:
            data_path: Path to data file
            config: Configuration dictionary
            tokenizer: Transformer tokenizer
        """
        super().__init__(data_path, config, tokenizer)
        self.window_context = config.get('window_context', 3)
    
    def __getitem__(self, idx: int) -> Dict:
        sample = super().__getitem__(idx)
        
        # Add additional processing for transformers
        sample['position_tensor'] = torch.tensor(sample['position'], dtype=torch.long)
        
        return sample


class XGBoostDataset:
    """Dataset wrapper for XGBoost models."""
    
    def __init__(self, data_path: str, config: Dict):
        """
        Initialize XGBoost dataset.
        
        Args:
            data_path: Path to data file
            config: Configuration dictionary
        """
        self.config = config
        
        # Load data
        if data_path.endswith('.csv'):
            self.data = pd.read_csv(data_path)
        elif data_path.endswith('.xlsx'):
            self.data = pd.read_excel(data_path)
        else:
            raise ValueError(f"Unsupported file format: {data_path}")
        
        # Initialize feature extractor
        self.feature_extractor = FeatureExtractor(config.get('window_size', 10))
        
        # Get column names from config
        self.sequence_col = config.get('sequence_col', 'Sequence')
        self.position_col = config.get('position_col', 'Position')
        self.target_col = config.get('target_col', 'target')
        self.protein_id_col = config.get('protein_id_col', 'Header')
        
        # Extract features if not already present
        if not self._has_features():
            self._extract_features()
    
    def _has_features(self) -> bool:
        """Check if features are already extracted."""
        # Look for feature columns (not ID or target columns)
        id_cols = {self.sequence_col, self.position_col, self.target_col, self.protein_id_col}
        feature_cols = [col for col in self.data.columns if col not in id_cols]
        return len(feature_cols) > 0
    
    def _extract_features(self) -> None:
        """Extract features for all samples."""
        features_list = []
        
        for _, row in self.data.iterrows():
            sequence = row[self.sequence_col]
            position = int(row[self.position_col])
            
            features = self.feature_extractor.extract_all_features(
                sequence, position, self.config
            )
            features_list.append(features)
        
        # Add features to dataframe
        feature_matrix = np.array(features_list)
        feature_names = [f"feature_{i}" for i in range(feature_matrix.shape[1])]
        
        for i, name in enumerate(feature_names):
            self.data[name] = feature_matrix[:, i]
    
    def get_features_and_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get feature matrix and labels."""
        # Identify feature columns
        id_cols = {self.sequence_col, self.position_col, self.target_col, self.protein_id_col}
        feature_cols = [col for col in self.data.columns if col not in id_cols]
        
        X = self.data[feature_cols].values
        y = self.data[self.target_col].values
        
        return X, y
    
    def get_protein_groups(self) -> np.ndarray:
        """Get protein groups for cross-validation."""
        return self.data[self.protein_id_col].values