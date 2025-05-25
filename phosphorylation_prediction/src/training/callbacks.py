"""Training callbacks for phosphorylation prediction."""

import os
import json
import time
from typing import Dict, Any, Optional, Union
from abc import ABC, abstractmethod
import logging


class Callback(ABC):
    """Base callback class."""
    
    def on_training_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of training."""
        pass
    
    def on_training_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of training."""
        pass
    
    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of each epoch."""
        pass
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each epoch."""
        pass
    
    def on_batch_start(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the start of each batch."""
        pass
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Called at the end of each batch."""
        pass


class EarlyStopping(Callback):
    """Early stopping callback with patience."""
    
    def __init__(self, patience: int = 5, metric: str = 'val_loss', mode: str = 'min',
                 min_delta: float = 0.0, restore_best_weights: bool = True,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize early stopping callback.
        
        Args:
            patience: Number of epochs with no improvement after which training will be stopped
            metric: Metric to monitor
            mode: 'min' or 'max' - whether the metric should be minimized or maximized
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore model weights from the best epoch
            logger: Optional logger instance
        """
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.logger = logger or logging.getLogger(__name__)
        
        self.best_score = None
        self.counter = 0
        self.best_weights = None
        self.stopped_epoch = 0
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best - min_delta
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda current, best: current > best + min_delta
            self.best_score = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Check for early stopping condition."""
        if logs is None:
            return
        
        current_score = logs.get(self.metric)
        if current_score is None:
            self.logger.warning(f"Metric '{self.metric}' not found in logs")
            return
        
        if self.is_better(current_score, self.best_score):
            self.best_score = current_score
            self.counter = 0
            
            # Store best weights if requested
            if self.restore_best_weights and hasattr(logs.get('model'), 'state_dict'):
                import copy
                self.best_weights = copy.deepcopy(logs['model'].state_dict())
        else:
            self.counter += 1
            
        if self.counter >= self.patience:
            self.stopped_epoch = epoch
            logs['stop_training'] = True
            
            # Restore best weights
            if self.restore_best_weights and self.best_weights is not None:
                model = logs.get('model')
                if model and hasattr(model, 'load_state_dict'):
                    model.load_state_dict(self.best_weights)
                    self.logger.info(f"Restored best weights from epoch {epoch - self.counter}")
            
            self.logger.info(
                f"Early stopping at epoch {epoch + 1}, "
                f"best {self.metric}: {self.best_score:.4f}"
            )


class ModelCheckpoint(Callback):
    """Model checkpointing callback."""
    
    def __init__(self, filepath: str, metric: str = 'val_loss', mode: str = 'min',
                 save_best_only: bool = True, save_frequency: int = 1,
                 logger: Optional[logging.Logger] = None):
        """
        Initialize model checkpoint callback.
        
        Args:
            filepath: Path to save model checkpoints
            metric: Metric to monitor for best model
            mode: 'min' or 'max' - whether the metric should be minimized or maximized
            save_best_only: Whether to save only the best model
            save_frequency: Frequency of saving (every N epochs)
            logger: Optional logger instance
        """
        self.filepath = filepath
        self.metric = metric
        self.mode = mode
        self.save_best_only = save_best_only
        self.save_frequency = save_frequency
        self.logger = logger or logging.getLogger(__name__)
        
        self.best_score = None
        
        if mode == 'min':
            self.is_better = lambda current, best: current < best
            self.best_score = float('inf')
        elif mode == 'max':
            self.is_better = lambda current, best: current > best
            self.best_score = float('-inf')
        else:
            raise ValueError(f"mode must be 'min' or 'max', got {mode}")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Save model checkpoint if conditions are met."""
        if logs is None:
            return
        
        model = logs.get('model')
        if model is None:
            self.logger.warning("No model found in logs for checkpointing")
            return
        
        # Check if we should save this epoch
        should_save = False
        
        if self.save_best_only:
            current_score = logs.get(self.metric)
            if current_score is not None and self.is_better(current_score, self.best_score):
                self.best_score = current_score
                should_save = True
        else:
            should_save = (epoch + 1) % self.save_frequency == 0
        
        if should_save:
            # Create filepath with epoch information
            if self.save_best_only:
                checkpoint_path = self.filepath.replace('.pt', f'_best.pt')
            else:
                checkpoint_path = self.filepath.replace('.pt', f'_epoch_{epoch+1}.pt')
            
            # Save model
            if hasattr(model, 'save'):
                model.save(checkpoint_path)
            else:
                # Fallback for PyTorch models
                import torch
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'metrics': logs
                }, checkpoint_path)
            
            self.logger.info(f"Model checkpoint saved: {checkpoint_path}")


class WandbCallback(Callback):
    """Weights & Biases logging callback."""
    
    def __init__(self, project: str, entity: Optional[str] = None, 
                 log_frequency: int = 1, log_predictions: bool = False,
                 tags: Optional[list] = None, logger: Optional[logging.Logger] = None):
        """
        Initialize WandB callback.
        
        Args:
            project: WandB project name
            entity: WandB entity (team/user)
            log_frequency: Frequency of logging (every N batches for batch logs)
            log_predictions: Whether to log prediction tables
            tags: List of tags for the run
            logger: Optional logger instance
        """
        self.project = project
        self.entity = entity
        self.log_frequency = log_frequency
        self.log_predictions = log_predictions
        self.tags = tags or []
        self.logger = logger or logging.getLogger(__name__)
        
        self.batch_count = 0
        self.wandb_initialized = False
        
        # Try to import wandb
        try:
            import wandb
            self.wandb = wandb
        except ImportError:
            self.logger.warning("wandb not installed, WandB logging disabled")
            self.wandb = None
    
    def on_training_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Initialize WandB run."""
        if self.wandb is None:
            return
        
        config = logs.get('config', {}) if logs else {}
        
        self.wandb.init(
            project=self.project,
            entity=self.entity,
            config=config,
            tags=self.tags,
            reinit=True
        )
        
        self.wandb_initialized = True
        self.logger.info(f"WandB run initialized: {self.wandb.run.url}")
    
    def on_training_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Finish WandB run."""
        if self.wandb and self.wandb_initialized:
            self.wandb.finish()
            self.wandb_initialized = False
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch metrics to WandB."""
        if not self.wandb_initialized or logs is None:
            return
        
        # Prepare metrics for logging
        metrics = {}
        for key, value in logs.items():
            if isinstance(value, (int, float)):
                metrics[key] = value
        
        # Add epoch information
        metrics['epoch'] = epoch
        
        self.wandb.log(metrics)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log batch metrics to WandB."""
        if not self.wandb_initialized or logs is None:
            return
        
        self.batch_count += 1
        
        # Log at specified frequency
        if self.batch_count % self.log_frequency == 0:
            # Prepare metrics for logging
            metrics = {}
            for key, value in logs.items():
                if isinstance(value, (int, float)):
                    metrics[f"batch_{key}"] = value
            
            # Add batch information
            metrics['batch'] = self.batch_count
            
            self.wandb.log(metrics)
            
            # Log predictions if requested
            if self.log_predictions and 'predictions' in logs and 'targets' in logs:
                self._log_predictions(logs['predictions'], logs['targets'])
    
    def _log_predictions(self, predictions, targets):
        """Log prediction table to WandB."""
        try:
            import pandas as pd
            
            # Create prediction table
            pred_df = pd.DataFrame({
                'predictions': predictions[:100],  # Limit to first 100 for performance
                'targets': targets[:100],
                'correct': (predictions[:100] == targets[:100])
            })
            
            # Log as wandb table
            table = self.wandb.Table(dataframe=pred_df)
            self.wandb.log({"predictions_sample": table})
            
        except Exception as e:
            self.logger.warning(f"Failed to log predictions: {e}")


class ProgressCallback(Callback):
    """Progress logging callback."""
    
    def __init__(self, log_frequency: int = 10, logger: Optional[logging.Logger] = None):
        """
        Initialize progress callback.
        
        Args:
            log_frequency: Frequency of progress logging
            logger: Optional logger instance
        """
        self.log_frequency = log_frequency
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.epoch_start_time = None
    
    def on_training_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log training start."""
        self.start_time = time.time()
        self.logger.info("Training started")
    
    def on_training_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log training completion."""
        if self.start_time:
            total_time = time.time() - self.start_time
            self.logger.info(f"Training completed in {total_time:.2f} seconds")
    
    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch start."""
        self.epoch_start_time = time.time()
        total_epochs = logs.get('total_epochs', '?') if logs else '?'
        self.logger.info(f"Epoch {epoch + 1}/{total_epochs}")
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Log epoch completion with metrics."""
        if self.epoch_start_time:
            epoch_time = time.time() - self.epoch_start_time
            
            # Format metrics for logging
            metrics_str = ""
            if logs:
                metric_parts = []
                for key, value in logs.items():
                    if isinstance(value, (int, float)) and not key.startswith('_'):
                        metric_parts.append(f"{key}: {value:.4f}")
                metrics_str = " - " + " - ".join(metric_parts)
            
            self.logger.info(f"Epoch {epoch + 1} completed in {epoch_time:.2f}s{metrics_str}")


class CallbackManager:
    """Manages multiple callbacks during training."""
    
    def __init__(self, callbacks: list = None):
        """
        Initialize callback manager.
        
        Args:
            callbacks: List of callback instances
        """
        self.callbacks = callbacks or []
    
    def add_callback(self, callback: Callback) -> None:
        """Add a callback to the manager."""
        self.callbacks.append(callback)
    
    def on_training_start(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_training_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_start(logs)
    
    def on_training_end(self, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_training_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_training_end(logs)
    
    def on_epoch_start(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_start(epoch, logs)
    
    def on_epoch_end(self, epoch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_epoch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_epoch_end(epoch, logs)
        
        # Check for early stopping
        return logs.get('stop_training', False)
    
    def on_batch_start(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_start for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_start(batch, logs)
    
    def on_batch_end(self, batch: int, logs: Optional[Dict[str, Any]] = None) -> None:
        """Call on_batch_end for all callbacks."""
        for callback in self.callbacks:
            callback.on_batch_end(batch, logs)