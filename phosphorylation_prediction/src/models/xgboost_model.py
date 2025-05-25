"""XGBoost model implementation for phosphorylation prediction."""

import numpy as np
import xgboost as xgb
from typing import Dict, Any, Optional, Union
import logging
import json
import os
from .base_model import BaseModel


class XGBoostModel(BaseModel):
    """XGBoost wrapper implementing BaseModel interface."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize XGBoost model.
        
        Args:
            config: Model configuration
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.feature_importance_ = None
        self.training_history = {}
        
        # Set default XGBoost parameters
        self.xgb_params = {
            'objective': 'binary:logistic',
            'eval_metric': config.get('eval_metric', ['logloss', 'auc']),
            'eta': config.get('learning_rate', 0.1),
            'max_depth': config.get('max_depth', 6),
            'min_child_weight': config.get('min_child_weight', 1),
            'subsample': config.get('subsample', 0.8),
            'colsample_bytree': config.get('colsample_bytree', 0.8),
            'tree_method': 'hist',
            'max_bin': 256,
            'random_state': config.get('seed', 42)
        }
        
        # Add GPU support if requested
        if config.get('use_gpu', False):
            self.xgb_params['device'] = 'cuda'
        
        self.num_boost_round = config.get('n_estimators', 1000)
        self.early_stopping_rounds = config.get('early_stopping_rounds', 50)
        
        self.logger.info(f"Initialized XGBoost model with parameters: {self.xgb_params}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, **kwargs) -> 'XGBoostModel':
        """
        Train XGBoost model with early stopping.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        self.validate_input(X_train, y_train)
        
        # Create DMatrix for training
        dtrain = xgb.DMatrix(X_train, label=y_train)
        
        # Set up evaluation sets
        evals = [(dtrain, 'train')]
        if X_val is not None and y_val is not None:
            self.validate_input(X_val, y_val)
            dval = xgb.DMatrix(X_val, label=y_val)
            evals.append((dval, 'validation'))
        
        # Initialize dictionary to store evaluation results
        evals_result = {}
        
        self.logger.info("Starting XGBoost training...")
        
        # Train model
        self.model = xgb.train(
            self.xgb_params,
            dtrain,
            num_boost_round=self.num_boost_round,
            evals=evals,
            early_stopping_rounds=self.early_stopping_rounds,
            evals_result=evals_result,
            verbose_eval=50
        )
        
        # Store training history
        self.training_history = evals_result
        
        # Get feature importance
        self.feature_importance_ = self.model.get_score(importance_type='gain')
        
        self.is_fitted = True
        self.logger.info("XGBoost training completed")
        
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return class predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Get probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.validate_input(X)
        
        dtest = xgb.DMatrix(X)
        probabilities = self.model.predict(dtest)
        
        return probabilities
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save XGBoost model
        model_path = path if path.endswith('.json') else f"{path}.json"
        self.model.save_model(model_path)
        
        # Save additional metadata
        metadata = {
            'config': self.config,
            'feature_importance': self.feature_importance_,
            'training_history': self.training_history,
            'xgb_params': self.xgb_params
        }
        
        metadata_path = model_path.replace('.json', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Metadata saved to {metadata_path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        # Load XGBoost model
        model_path = path if path.endswith('.json') else f"{path}.json"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        self.model = xgb.Booster()
        self.model.load_model(model_path)
        
        # Load metadata if available
        metadata_path = model_path.replace('.json', '_metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
            
            self.feature_importance_ = metadata.get('feature_importance', {})
            self.training_history = metadata.get('training_history', {})
            # Update config with saved parameters
            self.config.update(metadata.get('config', {}))
            self.xgb_params = metadata.get('xgb_params', self.xgb_params)
        
        self.is_fitted = True
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_feature_importance(self, importance_type: str = 'gain') -> Dict[str, float]:
        """
        Get feature importance scores.
        
        Args:
            importance_type: Type of importance ('gain', 'weight', 'cover')
            
        Returns:
            Feature importance dictionary
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get feature importance")
        
        if importance_type == 'gain' and self.feature_importance_ is not None:
            return self.feature_importance_
        
        return self.model.get_score(importance_type=importance_type)
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history including loss curves.
        
        Returns:
            Training history dictionary
        """
        return self.training_history
    
    def get_best_iteration(self) -> int:
        """
        Get the best iteration from early stopping.
        
        Returns:
            Best iteration number
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get best iteration")
        
        return self.model.best_iteration
    
    def get_num_features(self) -> int:
        """
        Get number of features the model was trained on.
        
        Returns:
            Number of features
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to get number of features")
        
        return self.model.num_features()
    
    def validate_input(self, X: np.ndarray, y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data for XGBoost.
        
        Args:
            X: Input features
            y: Optional target labels
            
        Raises:
            ValueError: If input is invalid
        """
        super().validate_input(X, y)
        
        # Additional XGBoost-specific validation
        if np.any(np.isnan(X)):
            self.logger.warning("Input contains NaN values")
        
        if np.any(np.isinf(X)):
            raise ValueError("Input contains infinite values")
    
    def __repr__(self) -> str:
        """String representation of the XGBoost model."""
        status = "fitted" if self.is_fitted else "unfitted"
        return f"XGBoostModel(status={status}, params={self.xgb_params})"