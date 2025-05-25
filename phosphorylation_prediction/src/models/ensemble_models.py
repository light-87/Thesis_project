"""Ensemble model implementations for phosphorylation prediction."""

import numpy as np
from typing import Dict, List, Any, Optional, Union, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, KFold
from sklearn.metrics import accuracy_score, f1_score
from scipy.optimize import minimize
import logging

from .base_model import BaseModel


class VotingEnsemble(BaseModel):
    """Voting ensemble with optimized weights."""
    
    def __init__(self, models: List[BaseModel], config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize voting ensemble.
        
        Args:
            models: List of base models
            config: Configuration dictionary
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.models = models
        self.weights = None
        self.voting_strategy = config.get('voting_strategy', 'soft')
        self.weight_optimization = config.get('voting_weights', 'optimize')
        
        # Validate models
        for model in self.models:
            if not hasattr(model, 'predict_proba'):
                raise ValueError("All models must support predict_proba for ensemble")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, **kwargs) -> 'VotingEnsemble':
        """
        Fit ensemble by optimizing voting weights.
        
        Args:
            X_train: Training features (not used directly, models should be pre-trained)
            y_train: Training labels (not used directly, models should be pre-trained)
            X_val: Validation features for weight optimization
            y_val: Validation labels for weight optimization
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if not all(model.is_fitted for model in self.models):
            raise ValueError("All base models must be fitted before ensemble training")
        
        if self.weight_optimization == 'equal':
            # Equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        elif self.weight_optimization == 'optimize' and X_val is not None and y_val is not None:
            # Optimize weights using validation data
            self.weights = self.optimize_weights(X_val, y_val)
        elif isinstance(self.weight_optimization, list):
            # Use provided weights
            if len(self.weight_optimization) != len(self.models):
                raise ValueError("Number of weights must match number of models")
            self.weights = np.array(self.weight_optimization)
            self.weights = self.weights / np.sum(self.weights)  # Normalize
        else:
            # Default to equal weights
            self.weights = np.ones(len(self.models)) / len(self.models)
        
        self.is_fitted = True
        self.logger.info(f"Voting ensemble fitted with weights: {self.weights}")
        
        return self
    
    def optimize_weights(self, X_val: np.ndarray, y_val: np.ndarray, 
                        metric: str = 'f1') -> np.ndarray:
        """
        Find optimal voting weights using validation data.
        
        Args:
            X_val: Validation features
            y_val: Validation labels
            metric: Metric to optimize ('accuracy' or 'f1')
            
        Returns:
            Optimal weights
        """
        # Get predictions from all models
        model_probs = []
        for model in self.models:
            probs = model.predict_proba(X_val)
            if probs.ndim == 2:
                probs = probs[:, 1]  # Take positive class probabilities
            model_probs.append(probs)
        
        model_probs = np.array(model_probs).T  # Shape: (n_samples, n_models)
        
        # Define objective function
        def objective(weights):
            weights = weights / np.sum(weights)  # Normalize
            ensemble_probs = np.dot(model_probs, weights)
            ensemble_preds = (ensemble_probs > 0.5).astype(int)
            
            if metric == 'accuracy':
                return -accuracy_score(y_val, ensemble_preds)
            elif metric == 'f1':
                return -f1_score(y_val, ensemble_preds)
            else:
                raise ValueError(f"Unsupported metric: {metric}")
        
        # Optimize weights
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        bounds = [(0, 1) for _ in range(len(self.models))]
        initial_weights = np.ones(len(self.models)) / len(self.models)
        
        result = minimize(
            objective, 
            initial_weights, 
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        
        return result.x
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return ensemble probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        if self.voting_strategy == 'soft':
            # Weighted average of probabilities
            ensemble_probs = np.zeros(len(X))
            for i, model in enumerate(self.models):
                probs = model.predict_proba(X)
                if probs.ndim == 2:
                    probs = probs[:, 1]  # Take positive class probabilities
                ensemble_probs += self.weights[i] * probs
        else:
            # Hard voting
            predictions = np.zeros((len(X), len(self.models)))
            for i, model in enumerate(self.models):
                predictions[:, i] = model.predict(X)
            
            # Weighted majority vote
            ensemble_probs = np.dot(predictions, self.weights)
        
        return ensemble_probs
    
    def save(self, path: str) -> None:
        """Save ensemble configuration."""
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        config = {
            'ensemble_type': 'voting',
            'config': self.config,
            'weights': self.weights.tolist() if self.weights is not None else None,
            'voting_strategy': self.voting_strategy
        }
        
        with open(f"{path}_ensemble.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load ensemble configuration."""
        import json
        
        with open(f"{path}_ensemble.json", 'r') as f:
            config = json.load(f)
        
        self.weights = np.array(config['weights']) if config['weights'] else None
        self.voting_strategy = config['voting_strategy']
        self.is_fitted = True


class StackingEnsemble(BaseModel):
    """Stacking ensemble with meta-learner."""
    
    def __init__(self, models: List[BaseModel], meta_learner: str, 
                 config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize stacking ensemble.
        
        Args:
            models: List of base models
            meta_learner: Type of meta-learner ('logistic_regression' or 'mlp')
            config: Configuration dictionary
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.models = models
        self.meta_learner_type = meta_learner
        self.use_probas = config.get('stacking_use_probas', True)
        self.cv_predictions = config.get('stacking_cv_predictions', True)
        
        # Initialize meta-learner
        if meta_learner == 'logistic_regression':
            self.meta_learner = LogisticRegression(random_state=42)
        elif meta_learner == 'mlp':
            self.meta_learner = MLPClassifier(
                hidden_layer_sizes=(10,),
                max_iter=1000,
                alpha=0.01,
                solver='adam',
                random_state=42
            )
        else:
            raise ValueError(f"Unsupported meta-learner: {meta_learner}")
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, **kwargs) -> 'StackingEnsemble':
        """
        Fit stacking ensemble with cross-validation.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (optional)
            y_val: Validation labels (optional)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if not all(model.is_fitted for model in self.models):
            raise ValueError("All base models must be fitted before ensemble training")
        
        # Create meta-features using cross-validation or validation set
        if self.cv_predictions and X_val is None:
            meta_features = self.create_cv_meta_features(X_train, y_train)
            meta_labels = y_train
        elif X_val is not None:
            meta_features = self.create_meta_features(X_val)
            meta_labels = y_val
        else:
            # Fallback: use training data (may lead to overfitting)
            meta_features = self.create_meta_features(X_train)
            meta_labels = y_train
            self.logger.warning("Using training data for meta-features may lead to overfitting")
        
        # Train meta-learner
        self.meta_learner.fit(meta_features, meta_labels)
        
        self.is_fitted = True
        self.logger.info(f"Stacking ensemble fitted with {self.meta_learner_type} meta-learner")
        
        return self
    
    def create_meta_features(self, X: np.ndarray) -> np.ndarray:
        """
        Generate meta-features from base models.
        
        Args:
            X: Input features
            
        Returns:
            Meta-features array
        """
        meta_features = []
        
        for model in self.models:
            if self.use_probas:
                probs = model.predict_proba(X)
                if probs.ndim == 2:
                    probs = probs[:, 1]  # Take positive class probabilities
                meta_features.append(probs)
            else:
                preds = model.predict(X)
                meta_features.append(preds)
        
        return np.column_stack(meta_features)
    
    def create_cv_meta_features(self, X: np.ndarray, y: np.ndarray, 
                               cv_folds: int = 5) -> np.ndarray:
        """
        Create meta-features using cross-validation to avoid overfitting.
        
        Args:
            X: Training features
            y: Training labels
            cv_folds: Number of CV folds
            
        Returns:
            Meta-features array
        """
        kf = KFold(n_splits=cv_folds, shuffle=True, random_state=42)
        meta_features = np.zeros((len(X), len(self.models)))
        
        for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
            X_fold_train, X_fold_val = X[train_idx], X[val_idx]
            y_fold_train, y_fold_val = y[train_idx], y[val_idx]
            
            # For each model, get predictions on validation fold
            for i, model in enumerate(self.models):
                # Clone and retrain model on fold training data
                # Note: This assumes models can be retrained
                # In practice, you might want to use pre-trained models
                
                if self.use_probas:
                    probs = model.predict_proba(X_fold_val)
                    if probs.ndim == 2:
                        probs = probs[:, 1]
                    meta_features[val_idx, i] = probs
                else:
                    preds = model.predict(X_fold_val)
                    meta_features[val_idx, i] = preds
        
        return meta_features
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return stacking ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        meta_features = self.create_meta_features(X)
        return self.meta_learner.predict(meta_features)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return stacking ensemble probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        meta_features = self.create_meta_features(X)
        probs = self.meta_learner.predict_proba(meta_features)
        
        if probs.ndim == 2:
            return probs[:, 1]  # Return positive class probabilities
        return probs
    
    def save(self, path: str) -> None:
        """Save stacking ensemble."""
        import pickle
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save meta-learner
        with open(f"{path}_meta_learner.pkl", 'wb') as f:
            pickle.dump(self.meta_learner, f)
        
        # Save configuration
        config = {
            'ensemble_type': 'stacking',
            'meta_learner_type': self.meta_learner_type,
            'config': self.config,
            'use_probas': self.use_probas,
            'cv_predictions': self.cv_predictions
        }
        
        with open(f"{path}_ensemble.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load stacking ensemble."""
        import pickle
        import json
        
        # Load meta-learner
        with open(f"{path}_meta_learner.pkl", 'rb') as f:
            self.meta_learner = pickle.load(f)
        
        # Load configuration
        with open(f"{path}_ensemble.json", 'r') as f:
            config = json.load(f)
        
        self.meta_learner_type = config['meta_learner_type']
        self.use_probas = config['use_probas']
        self.cv_predictions = config['cv_predictions']
        self.is_fitted = True


class DynamicEnsemble(BaseModel):
    """Dynamic model selection based on input similarity."""
    
    def __init__(self, models: List[BaseModel], config: Dict[str, Any], 
                 logger: Optional[logging.Logger] = None):
        """
        Initialize dynamic ensemble.
        
        Args:
            models: List of base models
            config: Configuration dictionary
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        self.models = models
        self.k_neighbors = config.get('dynamic_k_neighbors', 5)
        self.similarity_metric = config.get('dynamic_similarity_metric', 'cosine')
        self.competence_regions = None
        self.training_features = None
        self.training_labels = None
    
    def fit(self, X_train: np.ndarray, y_train: np.ndarray, 
            X_val: Optional[np.ndarray] = None, 
            y_val: Optional[np.ndarray] = None, **kwargs) -> 'DynamicEnsemble':
        """
        Fit dynamic ensemble by estimating competence regions.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features (if available)
            y_val: Validation labels (if available)
            **kwargs: Additional arguments
            
        Returns:
            Self for method chaining
        """
        if not all(model.is_fitted for model in self.models):
            raise ValueError("All base models must be fitted before ensemble training")
        
        # Use validation data if available, otherwise use training data
        if X_val is not None and y_val is not None:
            self.training_features = X_val
            self.training_labels = y_val
        else:
            self.training_features = X_train
            self.training_labels = y_train
        
        # Estimate competence for each model on training data
        self.competence_regions = self.estimate_competence(self.training_features)
        
        self.is_fitted = True
        self.logger.info("Dynamic ensemble fitted with competence estimation")
        
        return self
    
    def estimate_competence(self, X: np.ndarray) -> np.ndarray:
        """
        Estimate model competence for each input.
        
        Args:
            X: Input features
            
        Returns:
            Competence matrix (n_samples, n_models)
        """
        n_samples = len(X)
        n_models = len(self.models)
        competence = np.zeros((n_samples, n_models))
        
        # Get predictions from all models
        model_predictions = []
        for model in self.models:
            preds = model.predict(X)
            model_predictions.append(preds)
        
        model_predictions = np.array(model_predictions).T  # (n_samples, n_models)
        
        # For each sample, estimate competence based on local accuracy
        for i in range(n_samples):
            # Find k nearest neighbors
            neighbors = self._find_neighbors(X[i], X, self.k_neighbors)
            
            # Calculate local accuracy for each model
            for j in range(n_models):
                if len(neighbors) > 0:
                    neighbor_preds = model_predictions[neighbors, j]
                    neighbor_labels = self.training_labels[neighbors]
                    
                    # Local accuracy as competence measure
                    competence[i, j] = np.mean(neighbor_preds == neighbor_labels)
                else:
                    competence[i, j] = 0.5  # Default competence
        
        return competence
    
    def _find_neighbors(self, sample: np.ndarray, X: np.ndarray, k: int) -> np.ndarray:
        """
        Find k nearest neighbors of a sample.
        
        Args:
            sample: Query sample
            X: Training samples
            k: Number of neighbors
            
        Returns:
            Indices of nearest neighbors
        """
        if self.similarity_metric == 'cosine':
            # Cosine similarity
            similarities = np.dot(X, sample) / (np.linalg.norm(X, axis=1) * np.linalg.norm(sample))
            # Handle NaN values
            similarities = np.nan_to_num(similarities)
            neighbors = np.argsort(similarities)[-k:]
        elif self.similarity_metric == 'euclidean':
            # Euclidean distance
            distances = np.linalg.norm(X - sample, axis=1)
            neighbors = np.argsort(distances)[:k]
        else:
            raise ValueError(f"Unsupported similarity metric: {self.similarity_metric}")
        
        return neighbors
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Return dynamic ensemble predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted classes
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return dynamic ensemble probability predictions.
        
        Args:
            X: Input features
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Ensemble must be fitted before making predictions")
        
        # Estimate competence for new samples
        competence = self.estimate_competence(X)
        
        # Get predictions from all models
        model_probs = []
        for model in self.models:
            probs = model.predict_proba(X)
            if probs.ndim == 2:
                probs = probs[:, 1]  # Take positive class probabilities
            model_probs.append(probs)
        
        model_probs = np.array(model_probs).T  # (n_samples, n_models)
        
        # Weighted average based on competence
        ensemble_probs = np.zeros(len(X))
        for i in range(len(X)):
            weights = competence[i]
            weights = weights / np.sum(weights) if np.sum(weights) > 0 else np.ones(len(self.models)) / len(self.models)
            ensemble_probs[i] = np.dot(model_probs[i], weights)
        
        return ensemble_probs
    
    def save(self, path: str) -> None:
        """Save dynamic ensemble."""
        import pickle
        import json
        import os
        
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save training data and competence regions
        data = {
            'training_features': self.training_features,
            'training_labels': self.training_labels,
            'competence_regions': self.competence_regions
        }
        
        with open(f"{path}_dynamic_data.pkl", 'wb') as f:
            pickle.dump(data, f)
        
        # Save configuration
        config = {
            'ensemble_type': 'dynamic',
            'config': self.config,
            'k_neighbors': self.k_neighbors,
            'similarity_metric': self.similarity_metric
        }
        
        with open(f"{path}_ensemble.json", 'w') as f:
            json.dump(config, f, indent=2)
    
    def load(self, path: str) -> None:
        """Load dynamic ensemble."""
        import pickle
        import json
        
        # Load training data and competence regions
        with open(f"{path}_dynamic_data.pkl", 'rb') as f:
            data = pickle.load(f)
        
        self.training_features = data['training_features']
        self.training_labels = data['training_labels']
        self.competence_regions = data['competence_regions']
        
        # Load configuration
        with open(f"{path}_ensemble.json", 'r') as f:
            config = json.load(f)
        
        self.k_neighbors = config['k_neighbors']
        self.similarity_metric = config['similarity_metric']
        self.is_fitted = True