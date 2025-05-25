"""Base model interface for phosphorylation prediction."""

from abc import ABC, abstractmethod
import numpy as np
from typing import Dict, Any, Optional, Union
import logging


class BaseModel(ABC):
    """Abstract base class for all models."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize base model.
        
        Args:
            config: Model configuration
            logger: Optional logger instance
        """
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.model = None
        self.is_fitted = False
        
    @abstractmethod
    def fit(self, X_train: Union[np.ndarray, Any], y_train: np.ndarray, 
            X_val: Optional[Union[np.ndarray, Any]] = None, 
            y_val: Optional[np.ndarray] = None, **kwargs) -> 'BaseModel':
        """
        Train the model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        pass
    
    @abstractmethod
    def predict(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Return class predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        pass
    
    @abstractmethod
    def predict_proba(self, X: Union[np.ndarray, Any]) -> np.ndarray:
        """
        Return probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        pass
    
    @abstractmethod
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        pass
    
    @abstractmethod
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """
        Get model parameters.
        
        Returns:
            Model configuration parameters
        """
        return self.config.copy()
    
    def set_params(self, **params) -> 'BaseModel':
        """
        Set model parameters.
        
        Args:
            **params: Parameters to set
            
        Returns:
            Self for method chaining
        """
        for key, value in params.items():
            if key in self.config:
                self.config[key] = value
        return self
    
    def get_feature_importance(self) -> Optional[Dict[str, float]]:
        """
        Get feature importance if available.
        
        Returns:
            Feature importance dictionary or None
        """
        return None
    
    def validate_input(self, X: Union[np.ndarray, Any], 
                      y: Optional[np.ndarray] = None) -> None:
        """
        Validate input data.
        
        Args:
            X: Input features
            y: Optional target labels
            
        Raises:
            ValueError: If input is invalid
        """
        if isinstance(X, np.ndarray):
            if X.ndim != 2:
                raise ValueError(f"Expected 2D array, got {X.ndim}D array")
            if X.shape[0] == 0:
                raise ValueError("Empty input array")
        
        if y is not None:
            if isinstance(y, np.ndarray):
                if y.ndim != 1:
                    raise ValueError(f"Expected 1D target array, got {y.ndim}D array")
                if len(y) != len(X):
                    raise ValueError(f"X and y have different lengths: {len(X)} vs {len(y)}")
    
    def __repr__(self) -> str:
        """String representation of the model."""
        return f"{self.__class__.__name__}(config={self.config})"