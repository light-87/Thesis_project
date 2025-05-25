"""Reproducibility utilities for phosphorylation prediction."""

import os
import random
import numpy as np
from typing import Optional


def set_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.
    
    Args:
        seed: Random seed value
    """
    # Python random
    random.seed(seed)
    
    # NumPy
    np.random.seed(seed)
    
    # PyTorch (if available)
    try:
        import torch
        torch.manual_seed(seed)
        
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            
            # Additional CUDA settings for reproducibility
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
            
    except ImportError:
        pass
    
    # TensorFlow (if available)
    try:
        import tensorflow as tf
        tf.random.set_seed(seed)
        
        # For TensorFlow 1.x compatibility
        if hasattr(tf, 'set_random_seed'):
            tf.set_random_seed(seed)
            
    except ImportError:
        pass
    
    # Set environment variable for hash randomization
    os.environ['PYTHONHASHSEED'] = str(seed)


def ensure_reproducibility(seed: int) -> None:
    """
    Ensure reproducibility by setting all random seeds and deterministic behavior.
    
    Args:
        seed: Random seed value
    """
    set_seed(seed)
    
    # Additional PyTorch settings for maximum reproducibility
    try:
        import torch
        
        if torch.cuda.is_available():
            # Force deterministic algorithms
            torch.use_deterministic_algorithms(True, warn_only=True)
            
            # Set environment variable for CUDA deterministic operations
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
            
    except ImportError:
        pass
    except Exception:
        # If deterministic algorithms cause issues, just warn
        pass


def get_random_state(seed: int) -> np.random.RandomState:
    """
    Get a NumPy RandomState object with the specified seed.
    
    Args:
        seed: Random seed value
        
    Returns:
        NumPy RandomState object
    """
    return np.random.RandomState(seed)


class ReproducibleRandom:
    """Context manager for reproducible random operations."""
    
    def __init__(self, seed: int):
        """
        Initialize reproducible random context.
        
        Args:
            seed: Random seed value
        """
        self.seed = seed
        self.original_states = {}
    
    def __enter__(self):
        """Enter the context and save current random states."""
        # Save current Python random state
        self.original_states['python'] = random.getstate()
        
        # Save current NumPy random state
        self.original_states['numpy'] = np.random.get_state()
        
        # Save current PyTorch random state (if available)
        try:
            import torch
            self.original_states['torch'] = torch.get_rng_state()
            
            if torch.cuda.is_available():
                self.original_states['torch_cuda'] = torch.cuda.get_rng_state_all()
                
        except ImportError:
            pass
        
        # Set the new seed
        set_seed(self.seed)
        
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit the context and restore original random states."""
        # Restore Python random state
        if 'python' in self.original_states:
            random.setstate(self.original_states['python'])
        
        # Restore NumPy random state
        if 'numpy' in self.original_states:
            np.random.set_state(self.original_states['numpy'])
        
        # Restore PyTorch random state (if available)
        try:
            import torch
            
            if 'torch' in self.original_states:
                torch.set_rng_state(self.original_states['torch'])
            
            if 'torch_cuda' in self.original_states and torch.cuda.is_available():
                torch.cuda.set_rng_state_all(self.original_states['torch_cuda'])
                
        except ImportError:
            pass


def create_deterministic_dataloader(dataset, batch_size: int, shuffle: bool = True,
                                  seed: int = 42, **kwargs):
    """
    Create a deterministic DataLoader with reproducible shuffling.
    
    Args:
        dataset: Dataset to load
        batch_size: Batch size
        shuffle: Whether to shuffle data
        seed: Random seed for shuffling
        **kwargs: Additional DataLoader arguments
        
    Returns:
        DataLoader with deterministic behavior
    """
    try:
        import torch
        from torch.utils.data import DataLoader
        
        def seed_worker(worker_id):
            """Seed worker for deterministic behavior."""
            worker_seed = torch.initial_seed() % 2**32
            np.random.seed(worker_seed)
            random.seed(worker_seed)
        
        # Create generator with seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        
        # Create DataLoader with deterministic settings
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            generator=generator,
            worker_init_fn=seed_worker,
            **kwargs
        )
        
        return dataloader
        
    except ImportError:
        raise ImportError("PyTorch is required for deterministic DataLoader")


def hash_config(config: dict) -> str:
    """
    Create a deterministic hash of a configuration dictionary.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Hash string
    """
    import hashlib
    import json
    
    # Convert config to JSON string with sorted keys for deterministic hashing
    config_str = json.dumps(config, sort_keys=True, default=str)
    
    # Create hash
    hash_obj = hashlib.md5(config_str.encode())
    return hash_obj.hexdigest()


def save_random_state(filepath: str) -> None:
    """
    Save current random states to file.
    
    Args:
        filepath: Path to save random states
    """
    import pickle
    
    states = {
        'python': random.getstate(),
        'numpy': np.random.get_state()
    }
    
    # Add PyTorch states if available
    try:
        import torch
        states['torch'] = torch.get_rng_state()
        
        if torch.cuda.is_available():
            states['torch_cuda'] = torch.cuda.get_rng_state_all()
            
    except ImportError:
        pass
    
    # Save to file
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'wb') as f:
        pickle.dump(states, f)


def load_random_state(filepath: str) -> None:
    """
    Load random states from file.
    
    Args:
        filepath: Path to load random states from
    """
    import pickle
    
    with open(filepath, 'rb') as f:
        states = pickle.load(f)
    
    # Restore Python random state
    if 'python' in states:
        random.setstate(states['python'])
    
    # Restore NumPy random state
    if 'numpy' in states:
        np.random.set_state(states['numpy'])
    
    # Restore PyTorch random states if available
    try:
        import torch
        
        if 'torch' in states:
            torch.set_rng_state(states['torch'])
        
        if 'torch_cuda' in states and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(states['torch_cuda'])
            
    except ImportError:
        pass


class ReproducibilityManager:
    """Manager for reproducibility settings and state tracking."""
    
    def __init__(self, seed: int, output_dir: Optional[str] = None):
        """
        Initialize reproducibility manager.
        
        Args:
            seed: Random seed
            output_dir: Directory to save reproducibility information
        """
        self.seed = seed
        self.output_dir = output_dir
        
        # Ensure reproducibility
        ensure_reproducibility(seed)
        
        # Save initial state if output directory provided
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            self.save_initial_state()
    
    def save_initial_state(self) -> None:
        """Save initial random state and environment information."""
        if not self.output_dir:
            return
        
        # Save random state
        state_file = os.path.join(self.output_dir, 'random_state.pkl')
        save_random_state(state_file)
        
        # Save environment information
        env_info = self.get_environment_info()
        env_file = os.path.join(self.output_dir, 'environment.json')
        
        import json
        with open(env_file, 'w') as f:
            json.dump(env_info, f, indent=2, default=str)
    
    def get_environment_info(self) -> dict:
        """
        Get information about the current environment for reproducibility.
        
        Returns:
            Environment information dictionary
        """
        import platform
        import sys
        
        env_info = {
            'seed': self.seed,
            'python_version': sys.version,
            'platform': platform.platform(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        }
        
        # Add package versions
        packages_to_check = [
            'numpy', 'pandas', 'scikit-learn', 'xgboost', 
            'torch', 'transformers', 'matplotlib', 'seaborn'
        ]
        
        package_versions = {}
        for package in packages_to_check:
            try:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                package_versions[package] = version
            except ImportError:
                package_versions[package] = 'not installed'
        
        env_info['package_versions'] = package_versions
        
        # Add CUDA information if available
        try:
            import torch
            if torch.cuda.is_available():
                env_info['cuda_version'] = torch.version.cuda
                env_info['cudnn_version'] = torch.backends.cudnn.version()
                env_info['gpu_count'] = torch.cuda.device_count()
                env_info['gpu_names'] = [
                    torch.cuda.get_device_name(i) 
                    for i in range(torch.cuda.device_count())
                ]
            else:
                env_info['cuda_available'] = False
        except ImportError:
            pass
        
        return env_info
    
    def create_reproducible_split(self, data, test_size: float = 0.2, 
                                stratify=None, groups=None):
        """
        Create a reproducible train-test split.
        
        Args:
            data: Data to split
            test_size: Fraction of data for test set
            stratify: Stratification target
            groups: Group labels for group-based splitting
            
        Returns:
            Split data
        """
        with ReproducibleRandom(self.seed):
            if groups is not None:
                from sklearn.model_selection import GroupShuffleSplit
                splitter = GroupShuffleSplit(
                    n_splits=1, 
                    test_size=test_size, 
                    random_state=self.seed
                )
                train_idx, test_idx = next(splitter.split(data, stratify, groups))
                return train_idx, test_idx
            else:
                from sklearn.model_selection import train_test_split
                return train_test_split(
                    data, 
                    test_size=test_size, 
                    random_state=self.seed, 
                    stratify=stratify
                )
    
    def verify_reproducibility(self) -> bool:
        """
        Verify that the current state is reproducible.
        
        Returns:
            True if reproducible, False otherwise
        """
        # Test basic random operations
        with ReproducibleRandom(self.seed):
            python_random = [random.random() for _ in range(10)]
            numpy_random = np.random.random(10).tolist()
        
        # Test again with same seed
        with ReproducibleRandom(self.seed):
            python_random_2 = [random.random() for _ in range(10)]
            numpy_random_2 = np.random.random(10).tolist()
        
        # Check if results are identical
        python_match = python_random == python_random_2
        numpy_match = np.allclose(numpy_random, numpy_random_2)
        
        return python_match and numpy_match