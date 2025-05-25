"""Single model experiment for phosphorylation prediction."""

import os
import numpy as np
from typing import Dict, Any, Optional
from torch.utils.data import DataLoader

from .base_experiment import BaseExperiment
from ..data import load_raw_data, split_dataset_by_protein, create_data_loader
from ..data.dataset import XGBoostDataset, TransformerDataset
from ..models import XGBoostModel, TransformerModel
from ..training import create_trainer, EarlyStopping, ModelCheckpoint, WandbCallback
from ..evaluation import calculate_metrics, plot_results


class SingleModelExperiment(BaseExperiment):
    """Experiment for training and evaluating a single model."""
    
    def __init__(self, config_path: str, model_type: str, experiment_name: Optional[str] = None):
        """
        Initialize single model experiment.
        
        Args:
            config_path: Path to configuration file
            model_type: Type of model ('xgboost' or 'transformer')
            experiment_name: Optional experiment name
        """
        super().__init__(config_path, experiment_name)
        
        self.model_type = model_type
        self.model = None
        self.trainer = None
        
        # Data storage
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self.logger.info(f"Initialized single model experiment for {model_type}")
    
    def load_data(self) -> None:
        """Load and prepare data for the experiment."""
        self.logger.info("Loading data...")
        
        data_config = self.config.get('data', {})
        
        # Check if pre-split data exists
        data_dir = data_config.get('data_dir', 'data')
        train_file = os.path.join(data_dir, 'train_data.csv')
        val_file = os.path.join(data_dir, 'val_data.csv')
        test_file = os.path.join(data_dir, 'test_data.csv')
        
        if all(os.path.exists(f) for f in [train_file, val_file, test_file]):
            self.logger.info("Loading pre-split data...")
            
            if self.model_type == 'xgboost':
                self.train_data = XGBoostDataset(train_file, self.config)
                self.val_data = XGBoostDataset(val_file, self.config)
                self.test_data = XGBoostDataset(test_file, self.config)
            
            elif self.model_type == 'transformer':
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.get('transformer', {}).get('model_name', 'facebook/esm2_t6_8M_UR50D')
                )
                
                self.train_data = TransformerDataset(train_file, self.config, tokenizer)
                self.val_data = TransformerDataset(val_file, self.config, tokenizer)
                self.test_data = TransformerDataset(test_file, self.config, tokenizer)
        
        else:
            self.logger.info("Loading raw data and splitting...")
            
            # Load raw data
            sequence_file = data_config.get('sequence_file', 'Sequence_data.txt')
            labels_file = data_config.get('labels_file', 'labels.xlsx')
            
            df_final = load_raw_data(sequence_file, labels_file)
            
            # Split data
            train_df, val_df, test_df = split_dataset_by_protein(
                df_final,
                train_ratio=data_config.get('train_ratio', 0.7),
                val_ratio=data_config.get('val_ratio', 0.15),
                test_ratio=data_config.get('test_ratio', 0.15),
                random_seed=self.config.get('experiment', {}).get('seed', 42)
            )
            
            # Save split data for future use
            os.makedirs(data_dir, exist_ok=True)
            train_df.to_csv(train_file, index=False)
            val_df.to_csv(val_file, index=False)
            test_df.to_csv(test_file, index=False)
            
            # Create datasets
            if self.model_type == 'xgboost':
                self.train_data = XGBoostDataset(train_file, self.config)
                self.val_data = XGBoostDataset(val_file, self.config)
                self.test_data = XGBoostDataset(test_file, self.config)
            
            elif self.model_type == 'transformer':
                from transformers import AutoTokenizer
                tokenizer = AutoTokenizer.from_pretrained(
                    self.config.get('transformer', {}).get('model_name', 'facebook/esm2_t6_8M_UR50D')
                )
                
                self.train_data = TransformerDataset(train_file, self.config, tokenizer)
                self.val_data = TransformerDataset(val_file, self.config, tokenizer)
                self.test_data = TransformerDataset(test_file, self.config, tokenizer)
        
        self.logger.info("Data loading completed")
    
    def preprocess_data(self) -> None:
        """Preprocess the loaded data."""
        self.logger.info("Preprocessing data...")
        
        if self.model_type == 'transformer':
            # Create data loaders for transformer
            self.train_loader = create_data_loader(self.train_data, self.config, is_training=True)
            self.val_loader = create_data_loader(self.val_data, self.config, is_training=False)
            self.test_loader = create_data_loader(self.test_data, self.config, is_training=False)
            
            self.logger.info(f"Created data loaders - Train: {len(self.train_loader)} batches, "
                           f"Val: {len(self.val_loader)} batches, Test: {len(self.test_loader)} batches")
        
        elif self.model_type == 'xgboost':
            # Extract features and labels for XGBoost
            self.X_train, self.y_train = self.train_data.get_features_and_labels()
            self.X_val, self.y_val = self.val_data.get_features_and_labels()
            self.X_test, self.y_test = self.test_data.get_features_and_labels()
            
            self.logger.info(f"Extracted features - Train: {self.X_train.shape}, "
                           f"Val: {self.X_val.shape}, Test: {self.X_test.shape}")
        
        self.logger.info("Data preprocessing completed")
    
    def train_models(self) -> None:
        """Train the model."""
        self.logger.info(f"Training {self.model_type} model...")
        
        # Create model
        if self.model_type == 'xgboost':
            model_config = self.config.get('xgboost', {})
            self.model = XGBoostModel(model_config, self.logger)
            
            # Train XGBoost model
            self.model.fit(self.X_train, self.y_train, self.X_val, self.y_val)
            
        elif self.model_type == 'transformer':
            model_config = self.config.get('transformer', {})
            self.model = TransformerModel(model_config, self.logger)
            
            # Train transformer model
            self.model.fit(self.train_loader, self.val_loader)
        
        # Store trained model
        self.trained_models = {self.model_type: self.model}
        
        self.logger.info("Model training completed")
    
    def evaluate_models(self) -> None:
        """Evaluate the trained model."""
        self.logger.info("Evaluating model...")
        
        # Get test predictions
        if self.model_type == 'xgboost':
            y_pred = self.model.predict(self.X_test)
            y_proba = self.model.predict_proba(self.X_test)
            
            # Calculate metrics
            metrics = calculate_metrics(self.y_test, y_pred, y_proba)
            
            # Store results
            self.results['test_results'] = {
                self.model_type: {
                    **metrics,
                    'y_true': self.y_test,
                    'y_pred': y_pred,
                    'y_proba': y_proba,
                    'features': self.X_test
                }
            }
            
        elif self.model_type == 'transformer':
            y_pred = self.model.predict(self.test_loader)
            y_proba = self.model.predict_proba(self.test_loader)
            
            # Get true labels from test dataset
            y_true = []
            for batch in self.test_loader:
                y_true.extend(batch['target'].numpy())
            y_true = np.array(y_true)
            
            # Calculate metrics
            metrics = calculate_metrics(y_true, y_pred, y_proba)
            
            # Store results
            self.results['test_results'] = {
                self.model_type: {
                    **metrics,
                    'y_true': y_true,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                }
            }
        
        # Get feature importance if available
        if hasattr(self.model, 'get_feature_importance'):
            feature_importance = self.model.get_feature_importance()
            self.results['feature_importance'] = feature_importance
        
        # Get training history if available
        if hasattr(self.model, 'get_training_history'):
            training_history = self.model.get_training_history()
            self.results['training_history'] = training_history
        
        self.logger.info("Model evaluation completed")
        
        # Log key metrics
        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, (int, float)):
                self.logger.info(f"{metric_name}: {metric_value:.4f}")
    
    def analyze_results(self) -> None:
        """Analyze and visualize results."""
        super().analyze_results()
        
        self.logger.info("Creating visualizations...")
        
        try:
            # Create plots
            plots_dir = os.path.join(self.output_dir, "plots")
            plot_results(self.results, plots_dir)
            
            self.logger.info("Visualizations created successfully")
            
        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")
    
    def save_predictions(self) -> None:
        """Save detailed predictions for analysis."""
        if 'test_results' not in self.results:
            return
        
        try:
            for model_name, model_results in self.results['test_results'].items():
                predictions_df = {
                    'y_true': model_results['y_true'],
                    'y_pred': model_results['y_pred'],
                    'y_proba': model_results['y_proba']
                }
                
                # Add additional information if available
                if hasattr(self, 'test_data'):
                    if hasattr(self.test_data, 'data'):
                        # For DataFrame-based datasets
                        if 'Header' in self.test_data.data.columns:
                            predictions_df['protein_id'] = self.test_data.data['Header'].values
                        if 'Position' in self.test_data.data.columns:
                            predictions_df['position'] = self.test_data.data['Position'].values
                        if 'Sequence' in self.test_data.data.columns:
                            predictions_df['sequence'] = self.test_data.data['Sequence'].values
                
                # Save to CSV
                import pandas as pd
                df = pd.DataFrame(predictions_df)
                predictions_path = os.path.join(self.output_dir, "results", f"{model_name}_predictions.csv")
                df.to_csv(predictions_path, index=False)
                
                self.logger.info(f"Predictions saved to {predictions_path}")
                
        except Exception as e:
            self.logger.warning(f"Error saving predictions: {e}")
    
    def save_artifacts(self) -> None:
        """Save all artifacts including predictions."""
        # Save predictions first
        self.save_predictions()
        
        # Call parent method for other artifacts
        super().save_artifacts()


class XGBoostExperiment(SingleModelExperiment):
    """Convenience class for XGBoost experiments."""
    
    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        super().__init__(config_path, 'xgboost', experiment_name)


class TransformerExperiment(SingleModelExperiment):
    """Convenience class for Transformer experiments."""
    
    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        super().__init__(config_path, 'transformer', experiment_name)