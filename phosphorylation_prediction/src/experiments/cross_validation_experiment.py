"""Cross-validation experiment for phosphorylation prediction."""

import os
import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, List
from sklearn.model_selection import StratifiedGroupKFold

from .base_experiment import BaseExperiment
from ..data import load_raw_data, XGBoostDataset
from ..models import XGBoostModel, TransformerModel
from ..evaluation import calculate_metrics, calculate_cv_metrics


class CrossValidationExperiment(BaseExperiment):
    """Experiment for cross-validation evaluation."""
    
    def __init__(self, config_path: str, model_type: str = 'xgboost', 
                 experiment_name: Optional[str] = None):
        """
        Initialize cross-validation experiment.
        
        Args:
            config_path: Path to configuration file
            model_type: Type of model to evaluate ('xgboost' or 'transformer')
            experiment_name: Optional experiment name
        """
        super().__init__(config_path, experiment_name)
        
        self.model_type = model_type
        self.cv_results = []
        self.fold_models = {}
        
        # CV configuration
        cv_config = self.config.get('cross_validation', {})
        self.n_folds = cv_config.get('n_folds', 5)
        self.cv_strategy = cv_config.get('strategy', 'stratified_group')
        
        # Data storage
        self.df_data = None
        self.X_full = None
        self.y_full = None
        self.groups = None
        
        self.logger.info(f"Initialized CV experiment for {model_type} with {self.n_folds} folds")
    
    def load_data(self) -> None:
        """Load and prepare data for cross-validation."""
        self.logger.info("Loading data for cross-validation...")
        
        data_config = self.config.get('data', {})
        
        # Check if full dataset exists
        data_dir = data_config.get('data_dir', 'data')
        full_data_file = os.path.join(data_dir, 'full_data.csv')
        
        if os.path.exists(full_data_file):
            self.logger.info("Loading existing full dataset...")
            self.df_data = pd.read_csv(full_data_file)
        else:
            self.logger.info("Loading raw data and preparing full dataset...")
            
            # Load raw data
            sequence_file = data_config.get('sequence_file', 'Sequence_data.txt')
            labels_file = data_config.get('labels_file', 'labels.xlsx')
            
            self.df_data = load_raw_data(sequence_file, labels_file)
            
            # Save for future use
            os.makedirs(data_dir, exist_ok=True)
            self.df_data.to_csv(full_data_file, index=False)
        
        self.logger.info(f"Loaded {len(self.df_data)} samples for cross-validation")
    
    def preprocess_data(self) -> None:
        """Preprocess data for cross-validation."""
        self.logger.info("Preprocessing data for cross-validation...")
        
        if self.model_type == 'xgboost':
            # Create temporary dataset to extract features
            temp_file = 'temp_cv_data.csv'
            self.df_data.to_csv(temp_file, index=False)
            
            try:
                dataset = XGBoostDataset(temp_file, self.config)
                self.X_full, self.y_full = dataset.get_features_and_labels()
                self.groups = dataset.get_protein_groups()
                
                self.logger.info(f"Extracted features: {self.X_full.shape}")
                
            finally:
                # Clean up temporary file
                if os.path.exists(temp_file):
                    os.remove(temp_file)
        
        elif self.model_type == 'transformer':
            # For transformer, we'll need to handle data differently in each fold
            # Store the DataFrame for fold-wise processing
            self.groups = self.df_data['Header'].values
            self.y_full = self.df_data['target'].values
            
            self.logger.info("Prepared data for transformer cross-validation")
        
        self.logger.info("Data preprocessing completed")
    
    def create_cv_splits(self) -> List:
        """Create cross-validation splits."""
        self.logger.info(f"Creating {self.n_folds}-fold CV splits...")
        
        if self.cv_strategy == 'stratified_group':
            # Use StratifiedGroupKFold to maintain class balance while respecting protein groups
            cv = StratifiedGroupKFold(n_splits=self.n_folds, shuffle=True, random_state=42)
            splits = list(cv.split(self.X_full if self.X_full is not None else self.df_data, 
                                 self.y_full, self.groups))
        else:
            raise ValueError(f"Unsupported CV strategy: {self.cv_strategy}")
        
        # Log split statistics
        for fold, (train_idx, val_idx) in enumerate(splits):
            train_proteins = len(set(self.groups[train_idx]))
            val_proteins = len(set(self.groups[val_idx]))
            train_pos = np.sum(self.y_full[train_idx])
            val_pos = np.sum(self.y_full[val_idx])
            
            self.logger.info(
                f"Fold {fold + 1}: Train={len(train_idx)} samples ({train_proteins} proteins, "
                f"{train_pos} positive), Val={len(val_idx)} samples ({val_proteins} proteins, "
                f"{val_pos} positive)"
            )
        
        return splits
    
    def train_models(self) -> None:
        """Train models using cross-validation."""
        self.logger.info("Starting cross-validation training...")
        
        # Create CV splits
        cv_splits = self.create_cv_splits()
        
        # Run cross-validation
        for fold, (train_idx, val_idx) in enumerate(cv_splits):
            self.logger.info(f"Training fold {fold + 1}/{self.n_folds}...")
            
            try:
                fold_results = self._train_fold(fold, train_idx, val_idx)
                self.cv_results.append(fold_results)
                
                self.logger.info(
                    f"Fold {fold + 1} completed - "
                    f"Accuracy: {fold_results['accuracy']:.4f}, "
                    f"F1: {fold_results['f1']:.4f}"
                )
                
            except Exception as e:
                self.logger.error(f"Error in fold {fold + 1}: {e}")
                # Continue with other folds
                continue
        
        self.logger.info(f"Cross-validation completed. {len(self.cv_results)}/{self.n_folds} folds successful")
    
    def _train_fold(self, fold: int, train_idx: np.ndarray, val_idx: np.ndarray) -> Dict[str, float]:
        """Train and evaluate a single fold."""
        if self.model_type == 'xgboost':
            return self._train_xgboost_fold(fold, train_idx, val_idx)
        elif self.model_type == 'transformer':
            return self._train_transformer_fold(fold, train_idx, val_idx)
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
    
    def _train_xgboost_fold(self, fold: int, train_idx: np.ndarray, val_idx: np.ndarray) -> Dict[str, float]:
        """Train and evaluate XGBoost for one fold."""
        # Split data
        X_train, X_val = self.X_full[train_idx], self.X_full[val_idx]
        y_train, y_val = self.y_full[train_idx], self.y_full[val_idx]
        
        # Create and train model
        model_config = self.config.get('xgboost', {})
        model = XGBoostModel(model_config, self.logger)
        
        # Train model
        model.fit(X_train, y_train, X_val, y_val)
        
        # Store model for later analysis
        self.fold_models[f'fold_{fold}'] = model
        
        # Evaluate on validation set
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)
        
        # Calculate metrics
        metrics = calculate_metrics(y_val, y_pred, y_proba)
        
        # Add fold information
        metrics['fold'] = fold
        metrics['train_size'] = len(train_idx)
        metrics['val_size'] = len(val_idx)
        
        return metrics
    
    def _train_transformer_fold(self, fold: int, train_idx: np.ndarray, val_idx: np.ndarray) -> Dict[str, float]:
        """Train and evaluate Transformer for one fold."""
        from torch.utils.data import Subset
        from transformers import AutoTokenizer
        from ..data.dataset import TransformerDataset
        from ..data import create_data_loader
        
        # Create fold data files
        fold_train_df = self.df_data.iloc[train_idx]
        fold_val_df = self.df_data.iloc[val_idx]
        
        train_file = f'temp_fold_{fold}_train.csv'
        val_file = f'temp_fold_{fold}_val.csv'
        
        try:
            fold_train_df.to_csv(train_file, index=False)
            fold_val_df.to_csv(val_file, index=False)
            
            # Create datasets
            tokenizer = AutoTokenizer.from_pretrained(
                self.config.get('transformer', {}).get('model_name', 'facebook/esm2_t6_8M_UR50D')
            )
            
            train_dataset = TransformerDataset(train_file, self.config, tokenizer)
            val_dataset = TransformerDataset(val_file, self.config, tokenizer)
            
            # Create data loaders
            train_loader = create_data_loader(train_dataset, self.config, is_training=True)
            val_loader = create_data_loader(val_dataset, self.config, is_training=False)
            
            # Create and train model
            model_config = self.config.get('transformer', {})
            model = TransformerModel(model_config, self.logger)
            
            # Train model (with reduced epochs for CV)
            original_epochs = model_config.get('num_epochs', 10)
            model_config['num_epochs'] = max(1, original_epochs // 2)  # Reduce epochs for CV
            
            model.fit(train_loader, val_loader)
            
            # Store model for later analysis
            self.fold_models[f'fold_{fold}'] = model
            
            # Evaluate on validation set
            y_pred = model.predict(val_loader)
            y_proba = model.predict_proba(val_loader)
            
            # Get true labels
            y_val = fold_val_df['target'].values
            
            # Calculate metrics
            metrics = calculate_metrics(y_val, y_pred, y_proba)
            
            # Add fold information
            metrics['fold'] = fold
            metrics['train_size'] = len(train_idx)
            metrics['val_size'] = len(val_idx)
            
            return metrics
            
        finally:
            # Clean up temporary files
            for temp_file in [train_file, val_file]:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
    
    def evaluate_models(self) -> None:
        """Evaluate cross-validation results."""
        self.logger.info("Evaluating cross-validation results...")
        
        if not self.cv_results:
            self.logger.warning("No CV results to evaluate")
            return
        
        # Calculate CV statistics
        cv_stats = calculate_cv_metrics(self.cv_results)
        
        self.results['cv_results'] = {
            'individual_folds': self.cv_results,
            'cv_statistics': cv_stats,
            'n_folds': len(self.cv_results),
            'model_type': self.model_type
        }
        
        # Log summary statistics
        self.logger.info("Cross-Validation Results Summary:")
        for metric, stats in cv_stats.items():
            if isinstance(stats, dict) and 'mean' in stats:
                self.logger.info(
                    f"{metric}: {stats['mean']:.4f} ± {stats['std']:.4f} "
                    f"(range: {stats['min']:.4f} - {stats['max']:.4f})"
                )
        
        # Check for significant differences across folds
        self._analyze_fold_stability()
        
        self.logger.info("Cross-validation evaluation completed")
    
    def _analyze_fold_stability(self) -> None:
        """Analyze stability across folds."""
        if len(self.cv_results) < 3:
            return
        
        # Check coefficient of variation for key metrics
        key_metrics = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        stability_analysis = {}
        
        for metric in key_metrics:
            if metric in self.cv_results[0]:
                values = [fold_result[metric] for fold_result in self.cv_results]
                mean_val = np.mean(values)
                std_val = np.std(values)
                cv_coeff = std_val / mean_val if mean_val > 0 else float('inf')
                
                stability_analysis[metric] = {
                    'coefficient_of_variation': cv_coeff,
                    'is_stable': cv_coeff < 0.1  # Consider stable if CV < 10%
                }
                
                if cv_coeff > 0.2:  # Warn if CV > 20%
                    self.logger.warning(
                        f"High variability in {metric} across folds (CV = {cv_coeff:.3f})"
                    )
        
        self.results['fold_stability'] = stability_analysis
    
    def analyze_results(self) -> None:
        """Analyze cross-validation results."""
        super().analyze_results()
        
        self.logger.info("Creating cross-validation visualizations...")
        
        try:
            plots_dir = os.path.join(self.output_dir, "plots")
            self._create_cv_plots(plots_dir)
            
            self.logger.info("Cross-validation visualizations created")
            
        except Exception as e:
            self.logger.warning(f"Error creating CV visualizations: {e}")
    
    def _create_cv_plots(self, plots_dir: str) -> None:
        """Create cross-validation specific plots."""
        import matplotlib.pyplot as plt
        
        if not self.cv_results:
            return
        
        # Plot metric distributions across folds
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        available_metrics = [m for m in metrics_to_plot if m in self.cv_results[0]]
        
        if available_metrics:
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, metric in enumerate(available_metrics[:6]):
                values = [fold_result[metric] for fold_result in self.cv_results]
                folds = [f"Fold {fold_result['fold'] + 1}" for fold_result in self.cv_results]
                
                ax = axes[i]
                bars = ax.bar(folds, values, alpha=0.7)
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} Across Folds')
                ax.set_ylim(0, 1)
                
                # Add value labels on bars
                for bar, value in zip(bars, values):
                    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
                
                # Add mean line
                mean_val = np.mean(values)
                ax.axhline(y=mean_val, color='red', linestyle='--', alpha=0.7,
                          label=f'Mean: {mean_val:.3f}')
                ax.legend()
                
                plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
            
            # Hide unused subplots
            for i in range(len(available_metrics), 6):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, "cv_metrics_by_fold.png"), dpi=300, bbox_inches='tight')
            plt.close()
        
        # Plot metric distributions as box plots
        if len(self.cv_results) >= 3:
            fig, ax = plt.subplots(figsize=(12, 6))
            
            metric_data = []
            metric_names = []
            
            for metric in available_metrics:
                values = [fold_result[metric] for fold_result in self.cv_results]
                metric_data.append(values)
                metric_names.append(metric.capitalize())
            
            if metric_data:
                bp = ax.boxplot(metric_data, labels=metric_names, patch_artist=True)
                
                # Color the boxes
                colors = plt.cm.Set3(np.linspace(0, 1, len(metric_data)))
                for patch, color in zip(bp['boxes'], colors):
                    patch.set_facecolor(color)
                    patch.set_alpha(0.7)
                
                ax.set_ylabel('Score')
                ax.set_title('Cross-Validation Metric Distributions')
                ax.grid(True, alpha=0.3)
                
                plt.xticks(rotation=45, ha='right')
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "cv_metric_distributions.png"), dpi=300, bbox_inches='tight')
                plt.close()
    
    def get_best_fold_model(self) -> Any:
        """Get the model from the best performing fold."""
        if not self.cv_results or not self.fold_models:
            return None
        
        # Find best fold by F1 score
        best_fold_idx = max(range(len(self.cv_results)), 
                           key=lambda i: self.cv_results[i].get('f1', 0))
        
        best_fold_key = f'fold_{best_fold_idx}'
        return self.fold_models.get(best_fold_key)
    
    def save_fold_models(self) -> None:
        """Save models from each fold."""
        try:
            models_dir = os.path.join(self.output_dir, "models", "folds")
            os.makedirs(models_dir, exist_ok=True)
            
            for fold_name, model in self.fold_models.items():
                if hasattr(model, 'save'):
                    model_path = os.path.join(models_dir, fold_name)
                    model.save(model_path)
                    self.logger.info(f"Saved {fold_name} model")
            
        except Exception as e:
            self.logger.warning(f"Error saving fold models: {e}")
    
    def save_artifacts(self) -> None:
        """Save cross-validation specific artifacts."""
        # Save fold models
        self.save_fold_models()
        
        # Save detailed CV results
        try:
            cv_results_path = os.path.join(self.output_dir, "results", "cv_detailed_results.json")
            import json
            with open(cv_results_path, 'w') as f:
                json.dump(self.cv_results, f, indent=2, default=str)
            
            self.logger.info(f"Detailed CV results saved to {cv_results_path}")
            
        except Exception as e:
            self.logger.warning(f"Error saving detailed CV results: {e}")
        
        # Call parent method
        super().save_artifacts()