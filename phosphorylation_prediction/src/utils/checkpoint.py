"""Checkpointing utilities for phosphorylation prediction."""

import os
import json
import pickle
import torch
import joblib
import shutil
from pathlib import Path
from typing import Dict, Any, Optional, List, Union
from datetime import datetime
import warnings

class Checkpoint:
    """Container for model checkpoint data."""
    
    def __init__(self,
                 model_state: Optional[Dict[str, Any]] = None,
                 optimizer_state: Optional[Dict[str, Any]] = None,
                 epoch: int = 0,
                 best_metric: float = 0.0,
                 config: Optional[Dict[str, Any]] = None,
                 metadata: Optional[Dict[str, Any]] = None):
        """
        Initialize checkpoint.
        
        Args:
            model_state: Model state dictionary
            optimizer_state: Optimizer state dictionary
            epoch: Current epoch
            best_metric: Best validation metric achieved
            config: Model configuration
            metadata: Additional metadata
        """
        self.model_state = model_state or {}
        self.optimizer_state = optimizer_state or {}
        self.epoch = epoch
        self.best_metric = best_metric
        self.config = config or {}
        self.metadata = metadata or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary."""
        return {
            'model_state': self.model_state,
            'optimizer_state': self.optimizer_state,
            'epoch': self.epoch,
            'best_metric': self.best_metric,
            'config': self.config,
            'metadata': self.metadata,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Checkpoint':
        """Create checkpoint from dictionary."""
        checkpoint = cls()
        for key, value in data.items():
            setattr(checkpoint, key, value)
        return checkpoint


class CheckpointManager:
    """Manager for saving and loading model checkpoints."""
    
    def __init__(self,
                 checkpoint_dir: Union[str, Path],
                 max_checkpoints: int = 5,
                 save_best_only: bool = False):
        """
        Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            max_checkpoints: Maximum number of checkpoints to keep
            save_best_only: Whether to save only the best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.max_checkpoints = max_checkpoints
        self.save_best_only = save_best_only
        self.best_metric = float('-inf')
        self.checkpoint_history: List[Dict[str, Any]] = []
    
    def save_checkpoint(self,
                       checkpoint: Checkpoint,
                       filename: Optional[str] = None,
                       is_best: bool = False) -> str:
        """
        Save checkpoint to disk.
        
        Args:
            checkpoint: Checkpoint to save
            filename: Custom filename (optional)
            is_best: Whether this is the best checkpoint
        
        Returns:
            Path to saved checkpoint
        """
        if filename is None:
            filename = f"checkpoint_epoch_{checkpoint.epoch:04d}.pt"
        
        checkpoint_path = self.checkpoint_dir / filename
        
        # Save checkpoint
        torch.save(checkpoint.to_dict(), checkpoint_path)
        
        # Update history
        checkpoint_info = {
            'path': str(checkpoint_path),
            'epoch': checkpoint.epoch,
            'metric': checkpoint.best_metric,
            'timestamp': checkpoint.timestamp,
            'is_best': is_best
        }
        self.checkpoint_history.append(checkpoint_info)
        
        # Save best checkpoint separately
        if is_best:
            best_path = self.checkpoint_dir / "best_checkpoint.pt"
            shutil.copy2(checkpoint_path, best_path)
            self.best_metric = checkpoint.best_metric
        
        # Clean up old checkpoints
        if not self.save_best_only:
            self._cleanup_checkpoints()
        
        # Save checkpoint history
        self._save_history()
        
        return str(checkpoint_path)
    
    def load_checkpoint(self,
                       filename: Optional[str] = None,
                       load_best: bool = False) -> Optional[Checkpoint]:
        """
        Load checkpoint from disk.
        
        Args:
            filename: Specific checkpoint filename
            load_best: Whether to load the best checkpoint
        
        Returns:
            Loaded checkpoint or None if not found
        """
        if load_best:
            checkpoint_path = self.checkpoint_dir / "best_checkpoint.pt"
        elif filename:
            checkpoint_path = self.checkpoint_dir / filename
        else:
            # Load latest checkpoint
            checkpoint_path = self._get_latest_checkpoint()
        
        if not checkpoint_path or not checkpoint_path.exists():
            return None
        
        try:
            data = torch.load(checkpoint_path, map_location='cpu')
            return Checkpoint.from_dict(data)
        except Exception as e:
            warnings.warn(f"Failed to load checkpoint {checkpoint_path}: {e}")
            return None
    
    def list_checkpoints(self) -> List[Dict[str, Any]]:
        """List all available checkpoints."""
        return sorted(self.checkpoint_history, key=lambda x: x['epoch'])
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return len(self.checkpoint_history) > 0
    
    def get_best_metric(self) -> float:
        """Get the best metric achieved."""
        return self.best_metric
    
    def _get_latest_checkpoint(self) -> Optional[Path]:
        """Get path to the latest checkpoint."""
        if not self.checkpoint_history:
            return None
        
        latest = max(self.checkpoint_history, key=lambda x: x['epoch'])
        return Path(latest['path'])
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints beyond max_checkpoints."""
        if len(self.checkpoint_history) <= self.max_checkpoints:
            return
        
        # Sort by epoch and keep only the latest ones
        sorted_checkpoints = sorted(self.checkpoint_history, key=lambda x: x['epoch'])
        to_remove = sorted_checkpoints[:-self.max_checkpoints]
        
        for checkpoint_info in to_remove:
            checkpoint_path = Path(checkpoint_info['path'])
            if checkpoint_path.exists() and not checkpoint_info.get('is_best', False):
                try:
                    checkpoint_path.unlink()
                except OSError:
                    pass
            
            self.checkpoint_history.remove(checkpoint_info)
    
    def _save_history(self):
        """Save checkpoint history to disk."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        try:
            with open(history_path, 'w') as f:
                json.dump(self.checkpoint_history, f, indent=2)
        except Exception:
            pass
    
    def _load_history(self):
        """Load checkpoint history from disk."""
        history_path = self.checkpoint_dir / "checkpoint_history.json"
        if history_path.exists():
            try:
                with open(history_path, 'r') as f:
                    self.checkpoint_history = json.load(f)
                
                # Update best metric
                if self.checkpoint_history:
                    best_checkpoints = [c for c in self.checkpoint_history if c.get('is_best', False)]
                    if best_checkpoints:
                        self.best_metric = max(c['metric'] for c in best_checkpoints)
            except Exception:
                self.checkpoint_history = []


class ModelSaver:
    """Utilities for saving and loading different model types."""
    
    @staticmethod
    def save_pytorch_model(model: torch.nn.Module,
                          filepath: Union[str, Path],
                          config: Optional[Dict[str, Any]] = None):
        """Save PyTorch model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model_state_dict': model.state_dict(),
            'model_class': model.__class__.__name__,
            'config': config
        }
        
        torch.save(save_dict, filepath)
    
    @staticmethod
    def load_pytorch_model(model: torch.nn.Module,
                          filepath: Union[str, Path]) -> torch.nn.Module:
        """Load PyTorch model."""
        checkpoint = torch.load(filepath, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model
    
    @staticmethod
    def save_sklearn_model(model: Any,
                          filepath: Union[str, Path],
                          config: Optional[Dict[str, Any]] = None):
        """Save scikit-learn model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'model': model,
            'config': config,
            'model_class': model.__class__.__name__
        }
        
        joblib.dump(save_dict, filepath)
    
    @staticmethod
    def load_sklearn_model(filepath: Union[str, Path]) -> Any:
        """Load scikit-learn model."""
        data = joblib.load(filepath)
        return data['model']
    
    @staticmethod
    def save_ensemble_model(models: Dict[str, Any],
                           weights: Optional[Dict[str, float]],
                           filepath: Union[str, Path],
                           config: Optional[Dict[str, Any]] = None):
        """Save ensemble model."""
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        save_dict = {
            'models': models,
            'weights': weights,
            'config': config,
            'ensemble_type': 'custom'
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(save_dict, f)
    
    @staticmethod
    def load_ensemble_model(filepath: Union[str, Path]) -> Dict[str, Any]:
        """Load ensemble model."""
        with open(filepath, 'rb') as f:
            return pickle.load(f)


def create_experiment_checkpoint(experiment_dir: Union[str, Path],
                               models: Dict[str, Any],
                               metrics: Dict[str, Any],
                               config: Dict[str, Any],
                               metadata: Optional[Dict[str, Any]] = None) -> str:
    """
    Create a complete experiment checkpoint.
    
    Args:
        experiment_dir: Experiment directory
        models: Dictionary of trained models
        metrics: Evaluation metrics
        config: Experiment configuration
        metadata: Additional metadata
    
    Returns:
        Path to experiment checkpoint
    """
    experiment_dir = Path(experiment_dir)
    checkpoint_dir = experiment_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Save models
    model_paths = {}
    for name, model in models.items():
        model_path = checkpoint_dir / f"{name}_model.pkl"
        
        if hasattr(model, 'state_dict'):  # PyTorch model
            ModelSaver.save_pytorch_model(model, model_path)
        else:  # Scikit-learn model
            ModelSaver.save_sklearn_model(model, model_path)
        
        model_paths[name] = str(model_path)
    
    # Create experiment checkpoint
    experiment_checkpoint = {
        'model_paths': model_paths,
        'metrics': metrics,
        'config': config,
        'metadata': metadata or {},
        'timestamp': datetime.now().isoformat()
    }
    
    # Save experiment checkpoint
    checkpoint_path = checkpoint_dir / "experiment_checkpoint.json"
    with open(checkpoint_path, 'w') as f:
        json.dump(experiment_checkpoint, f, indent=2)
    
    return str(checkpoint_path)


def load_experiment_checkpoint(checkpoint_path: Union[str, Path]) -> Dict[str, Any]:
    """
    Load complete experiment checkpoint.
    
    Args:
        checkpoint_path: Path to experiment checkpoint
    
    Returns:
        Experiment data including models, metrics, and config
    """
    with open(checkpoint_path, 'r') as f:
        checkpoint_data = json.load(f)
    
    # Load models
    models = {}
    for name, model_path in checkpoint_data['model_paths'].items():
        model_path = Path(model_path)
        
        if model_path.suffix == '.pkl':
            # Try to load as scikit-learn model first
            try:
                models[name] = ModelSaver.load_sklearn_model(model_path)
            except:
                # If that fails, try as ensemble
                models[name] = ModelSaver.load_ensemble_model(model_path)
        else:
            # PyTorch model - need model class to load properly
            warnings.warn(f"Cannot auto-load PyTorch model {name}. "
                         f"Manual loading required with model class.")
    
    checkpoint_data['models'] = models
    return checkpoint_data