"""Training classes for phosphorylation prediction models."""

import time
import logging
from typing import Dict, List, Any, Optional, Union, Type
from abc import ABC, abstractmethod
from torch.utils.data import DataLoader

from .callbacks import CallbackManager, Callback
from ..models.base_model import BaseModel


class BaseTrainer(ABC):
    """Base trainer class with callback integration."""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize base trainer.
        
        Args:
            model: Model to train
            config: Training configuration
            logger: Optional logger instance
        """
        self.model = model
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.callback_manager = CallbackManager()
        
        # Training state
        self.current_epoch = 0
        self.total_epochs = config.get('num_epochs', 10)
        self.training_history = {'train': [], 'val': []}
        
    def add_callback(self, callback: Callback) -> None:
        """
        Add training callback.
        
        Args:
            callback: Callback instance to add
        """
        self.callback_manager.add_callback(callback)
    
    def add_callbacks(self, callbacks: List[Callback]) -> None:
        """
        Add multiple training callbacks.
        
        Args:
            callbacks: List of callback instances
        """
        for callback in callbacks:
            self.add_callback(callback)
    
    @abstractmethod
    def train_epoch(self, train_data) -> Dict[str, float]:
        """
        Train for one epoch.
        
        Args:
            train_data: Training data
            
        Returns:
            Training metrics for the epoch
        """
        pass
    
    @abstractmethod
    def validate_epoch(self, val_data) -> Dict[str, float]:
        """
        Validate for one epoch.
        
        Args:
            val_data: Validation data
            
        Returns:
            Validation metrics for the epoch
        """
        pass
    
    def train(self, train_data, val_data: Optional = None) -> BaseModel:
        """
        Main training loop with callback integration.
        
        Args:
            train_data: Training data
            val_data: Validation data (optional)
            
        Returns:
            Trained model
        """
        # Prepare logs for callbacks
        logs = {
            'model': self.model,
            'config': self.config,
            'total_epochs': self.total_epochs
        }
        
        # Start training
        self.callback_manager.on_training_start(logs)
        
        try:
            for epoch in range(self.total_epochs):
                self.current_epoch = epoch
                
                # Epoch start
                epoch_logs = logs.copy()
                epoch_logs.update({'epoch': epoch})
                self.callback_manager.on_epoch_start(epoch, epoch_logs)
                
                # Train epoch
                train_metrics = self.train_epoch(train_data)
                self.training_history['train'].append(train_metrics)
                
                # Validation epoch
                val_metrics = {}
                if val_data is not None:
                    val_metrics = self.validate_epoch(val_data)
                    self.training_history['val'].append(val_metrics)
                
                # Combine metrics
                all_metrics = {}
                for key, value in train_metrics.items():
                    all_metrics[f'train_{key}'] = value
                for key, value in val_metrics.items():
                    all_metrics[f'val_{key}'] = value
                
                # Epoch end
                epoch_logs.update(all_metrics)
                epoch_logs['model'] = self.model
                
                # Check for early stopping
                stop_training = self.callback_manager.on_epoch_end(epoch, epoch_logs)
                if stop_training:
                    self.logger.info(f"Training stopped early at epoch {epoch + 1}")
                    break
                
        except KeyboardInterrupt:
            self.logger.info("Training interrupted by user")
        except Exception as e:
            self.logger.error(f"Training failed with error: {e}")
            raise
        finally:
            # End training
            self.callback_manager.on_training_end(logs)
        
        return self.model
    
    def get_training_history(self) -> Dict[str, List[Dict[str, float]]]:
        """
        Get training history.
        
        Returns:
            Training history dictionary
        """
        return self.training_history


class XGBoostTrainer(BaseTrainer):
    """Trainer for XGBoost models."""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize XGBoost trainer."""
        super().__init__(model, config, logger)
        
        # XGBoost doesn't have epochs in the traditional sense
        # We'll simulate epochs by training incrementally
        self.trees_per_epoch = config.get('trees_per_epoch', 100)
        self.total_trees = config.get('n_estimators', 1000)
        self.total_epochs = max(1, self.total_trees // self.trees_per_epoch)
    
    def train_epoch(self, train_data) -> Dict[str, float]:
        """
        Train XGBoost for one epoch (batch of trees).
        
        Args:
            train_data: Training data tuple (X_train, y_train)
            
        Returns:
            Training metrics
        """
        # For XGBoost, we don't train epoch by epoch in the traditional sense
        # This is a placeholder that would need adaptation based on the specific
        # XGBoost training approach
        
        if self.current_epoch == 0:
            # Train the full model on first epoch
            X_train, y_train = train_data
            self.model.fit(X_train, y_train)
        
        # Return dummy metrics (in practice, you'd extract from XGBoost)
        return {
            'loss': 0.5,  # Placeholder
            'accuracy': 0.8  # Placeholder
        }
    
    def validate_epoch(self, val_data) -> Dict[str, float]:
        """
        Validate XGBoost model.
        
        Args:
            val_data: Validation data tuple (X_val, y_val)
            
        Returns:
            Validation metrics
        """
        X_val, y_val = val_data
        
        # Get predictions
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'loss': 0.0,  # XGBoost doesn't provide loss directly
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_proba)
        }
        
        return metrics


class TransformerTrainer(BaseTrainer):
    """Trainer for Transformer models."""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize Transformer trainer."""
        super().__init__(model, config, logger)
        self.batch_count = 0
    
    def train_epoch(self, train_data: DataLoader) -> Dict[str, float]:
        """
        Train Transformer for one epoch.
        
        Args:
            train_data: Training DataLoader
            
        Returns:
            Training metrics
        """
        # The actual training logic is handled by the TransformerModel
        # Here we just need to call the model's training method
        
        # This is a simplified version - the full implementation would
        # handle the training loop with proper batch processing
        
        return {
            'loss': 0.3,  # Placeholder - would come from actual training
            'accuracy': 0.85,  # Placeholder
            'f1': 0.82  # Placeholder
        }
    
    def validate_epoch(self, val_data: DataLoader) -> Dict[str, float]:
        """
        Validate Transformer model.
        
        Args:
            val_data: Validation DataLoader
            
        Returns:
            Validation metrics
        """
        # Similar to training, this would use the model's evaluation method
        
        return {
            'loss': 0.35,  # Placeholder
            'accuracy': 0.83,  # Placeholder
            'f1': 0.80,  # Placeholder
            'auc': 0.88  # Placeholder
        }


class EnsembleTrainer(BaseTrainer):
    """Trainer for ensemble models."""
    
    def __init__(self, model: BaseModel, config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """Initialize ensemble trainer."""
        super().__init__(model, config, logger)
        
        # Ensemble training is typically simpler - just optimization of weights
        self.total_epochs = config.get('ensemble_epochs', 1)
    
    def train_epoch(self, train_data) -> Dict[str, float]:
        """
        Train ensemble for one epoch.
        
        Args:
            train_data: Training data (depends on ensemble type)
            
        Returns:
            Training metrics
        """
        if self.current_epoch == 0:
            # Train ensemble on first epoch
            if hasattr(train_data, '__len__') and len(train_data) == 2:
                X_train, y_train = train_data
                self.model.fit(X_train, y_train)
        
        return {
            'loss': 0.0,  # Ensemble doesn't have traditional loss
            'accuracy': 0.86  # Placeholder
        }
    
    def validate_epoch(self, val_data) -> Dict[str, float]:
        """
        Validate ensemble model.
        
        Args:
            val_data: Validation data
            
        Returns:
            Validation metrics
        """
        X_val, y_val = val_data
        
        # Get predictions
        y_pred = self.model.predict(X_val)
        y_proba = self.model.predict_proba(X_val)
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'f1': f1_score(y_val, y_pred, zero_division=0),
            'auc': roc_auc_score(y_val, y_proba)
        }
        
        return metrics


def create_trainer(model_type: str, model: BaseModel, config: Dict[str, Any], 
                  logger: Optional[logging.Logger] = None) -> BaseTrainer:
    """
    Factory function for creating appropriate trainer.
    
    Args:
        model_type: Type of model ('xgboost', 'transformer', 'ensemble')
        model: Model instance
        config: Training configuration
        logger: Optional logger instance
        
    Returns:
        Appropriate trainer instance
    """
    trainer_map = {
        'xgboost': XGBoostTrainer,
        'transformer': TransformerTrainer,
        'ensemble': EnsembleTrainer
    }
    
    if model_type not in trainer_map:
        raise ValueError(f"Unsupported model type: {model_type}")
    
    trainer_class = trainer_map[model_type]
    return trainer_class(model, config, logger)


class TrainingSession:
    """High-level training session manager."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize training session.
        
        Args:
            config: Training configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.start_time = None
        self.end_time = None
    
    def run_training(self, model: BaseModel, train_data, val_data: Optional = None,
                    callbacks: Optional[List[Callback]] = None) -> BaseModel:
        """
        Run a complete training session.
        
        Args:
            model: Model to train
            train_data: Training data
            val_data: Validation data (optional)
            callbacks: List of callbacks (optional)
            
        Returns:
            Trained model
        """
        self.start_time = time.time()
        
        # Determine model type
        model_type = self._determine_model_type(model)
        
        # Create trainer
        trainer = create_trainer(model_type, model, self.config, self.logger)
        
        # Add callbacks
        if callbacks:
            trainer.add_callbacks(callbacks)
        
        # Run training
        try:
            trained_model = trainer.train(train_data, val_data)
            self.end_time = time.time()
            
            # Log session summary
            duration = self.end_time - self.start_time
            self.logger.info(f"Training session completed in {duration:.2f} seconds")
            
            return trained_model
            
        except Exception as e:
            self.end_time = time.time()
            self.logger.error(f"Training session failed: {e}")
            raise
    
    def _determine_model_type(self, model: BaseModel) -> str:
        """
        Determine model type from model class.
        
        Args:
            model: Model instance
            
        Returns:
            Model type string
        """
        class_name = model.__class__.__name__.lower()
        
        if 'xgboost' in class_name:
            return 'xgboost'
        elif 'transformer' in class_name:
            return 'transformer'
        elif 'ensemble' in class_name:
            return 'ensemble'
        else:
            # Default to generic trainer
            return 'xgboost'  # Fallback
    
    def get_session_duration(self) -> Optional[float]:
        """
        Get training session duration.
        
        Returns:
            Duration in seconds or None if session not completed
        """
        if self.start_time and self.end_time:
            return self.end_time - self.start_time
        return None