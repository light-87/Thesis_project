"""Ensemble experiment for phosphorylation prediction."""

import os
import numpy as np
from typing import Dict, Any, Optional, List

from .base_experiment import BaseExperiment
from .single_model_experiment import XGBoostExperiment, TransformerExperiment
from ..models.ensemble_models import VotingEnsemble, StackingEnsemble, DynamicEnsemble
from ..evaluation import calculate_metrics, compare_models, plot_results


class EnsembleExperiment(BaseExperiment):
    """Experiment for ensemble methods."""
    
    def __init__(self, config_path: str, ensemble_types: Optional[List[str]] = None, 
                 experiment_name: Optional[str] = None):
        """
        Initialize ensemble experiment.
        
        Args:
            config_path: Path to configuration file
            ensemble_types: List of ensemble types to train ['voting', 'stacking', 'dynamic']
            experiment_name: Optional experiment name
        """
        super().__init__(config_path, experiment_name)
        
        self.ensemble_types = ensemble_types or ['voting', 'stacking', 'dynamic']
        self.base_models = {}
        self.ensemble_models = {}
        
        # Data storage (will be loaded from base model experiments)
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        self.logger.info(f"Initialized ensemble experiment with types: {self.ensemble_types}")
    
    def load_data(self) -> None:
        """Load data by training base models."""
        self.logger.info("Loading data via base model training...")
        
        # Train base models to get data and predictions
        self.train_base_models()
        
        self.logger.info("Data loaded via base models")
    
    def preprocess_data(self) -> None:
        """Preprocess data - already handled by base models."""
        self.logger.info("Data preprocessing completed by base models")
    
    def train_base_models(self) -> None:
        """Train base models (XGBoost and Transformer)."""
        self.logger.info("Training base models...")
        
        try:
            # Train XGBoost model
            self.logger.info("Training XGBoost base model...")
            xgb_experiment = XGBoostExperiment(
                self.config_path, 
                experiment_name=f"{self.experiment_name}_xgboost_base"
            )
            xgb_results = xgb_experiment.run()
            
            # Store XGBoost model and data
            self.base_models['xgboost'] = xgb_experiment.model
            
            # Get data from XGBoost experiment
            self.X_train = xgb_experiment.X_train
            self.y_train = xgb_experiment.y_train
            self.X_val = xgb_experiment.X_val
            self.y_val = xgb_experiment.y_val
            self.X_test = xgb_experiment.X_test
            self.y_test = xgb_experiment.y_test
            
            self.logger.info("XGBoost base model trained successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to train XGBoost base model: {e}")
            raise
        
        try:
            # Train Transformer model
            self.logger.info("Training Transformer base model...")
            transformer_experiment = TransformerExperiment(
                self.config_path,
                experiment_name=f"{self.experiment_name}_transformer_base"
            )
            transformer_results = transformer_experiment.run()
            
            # Store Transformer model
            self.base_models['transformer'] = transformer_experiment.model
            
            self.logger.info("Transformer base model trained successfully")
            
        except Exception as e:
            self.logger.warning(f"Failed to train Transformer base model: {e}")
            # Continue with just XGBoost if Transformer fails
            if 'xgboost' not in self.base_models:
                raise
        
        self.logger.info(f"Base model training completed. Available models: {list(self.base_models.keys())}")
    
    def train_models(self) -> None:
        """Train ensemble models."""
        self.logger.info("Training ensemble models...")
        
        if len(self.base_models) < 2:
            self.logger.warning("Need at least 2 base models for ensemble. Skipping ensemble training.")
            return
        
        base_model_list = list(self.base_models.values())
        ensemble_config = self.config.get('ensemble', {})
        
        # Train different ensemble types
        for ensemble_type in self.ensemble_types:
            try:
                self.logger.info(f"Training {ensemble_type} ensemble...")
                
                if ensemble_type == 'voting':
                    ensemble_model = VotingEnsemble(
                        models=base_model_list,
                        config=ensemble_config,
                        logger=self.logger
                    )
                    
                    # Fit voting ensemble
                    ensemble_model.fit(
                        X_train=self.X_train,
                        y_train=self.y_train,
                        X_val=self.X_val,
                        y_val=self.y_val
                    )
                
                elif ensemble_type == 'stacking':
                    meta_learner = ensemble_config.get('stacking_meta_learner', 'logistic_regression')
                    
                    ensemble_model = StackingEnsemble(
                        models=base_model_list,
                        meta_learner=meta_learner,
                        config=ensemble_config,
                        logger=self.logger
                    )
                    
                    # Fit stacking ensemble
                    ensemble_model.fit(
                        X_train=self.X_train,
                        y_train=self.y_train,
                        X_val=self.X_val,
                        y_val=self.y_val
                    )
                
                elif ensemble_type == 'dynamic':
                    ensemble_model = DynamicEnsemble(
                        models=base_model_list,
                        config=ensemble_config,
                        logger=self.logger
                    )
                    
                    # Fit dynamic ensemble
                    ensemble_model.fit(
                        X_train=self.X_train,
                        y_train=self.y_train,
                        X_val=self.X_val,
                        y_val=self.y_val
                    )
                
                else:
                    self.logger.warning(f"Unknown ensemble type: {ensemble_type}")
                    continue
                
                self.ensemble_models[ensemble_type] = ensemble_model
                self.logger.info(f"{ensemble_type} ensemble trained successfully")
                
            except Exception as e:
                self.logger.error(f"Failed to train {ensemble_type} ensemble: {e}")
                continue
        
        # Store all trained models
        self.trained_models = {**self.base_models, **self.ensemble_models}
        
        self.logger.info(f"Ensemble training completed. Trained models: {list(self.trained_models.keys())}")
    
    def evaluate_models(self) -> None:
        """Evaluate all models (base and ensemble)."""
        self.logger.info("Evaluating all models...")
        
        self.results['test_results'] = {}
        
        # Evaluate base models
        for model_name, model in self.base_models.items():
            try:
                self.logger.info(f"Evaluating {model_name} model...")
                
                y_pred = model.predict(self.X_test)
                y_proba = model.predict_proba(self.X_test)
                
                # Handle different probability formats
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]
                
                metrics = calculate_metrics(self.y_test, y_pred, y_proba)
                
                self.results['test_results'][model_name] = {
                    **metrics,
                    'y_true': self.y_test,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                }
                
                self.logger.info(f"{model_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {model_name}: {e}")
                continue
        
        # Evaluate ensemble models
        for ensemble_name, ensemble_model in self.ensemble_models.items():
            try:
                self.logger.info(f"Evaluating {ensemble_name} ensemble...")
                
                y_pred = ensemble_model.predict(self.X_test)
                y_proba = ensemble_model.predict_proba(self.X_test)
                
                # Handle different probability formats
                if y_proba.ndim == 2:
                    y_proba = y_proba[:, 1]
                
                metrics = calculate_metrics(self.y_test, y_pred, y_proba)
                
                self.results['test_results'][ensemble_name] = {
                    **metrics,
                    'y_true': self.y_test,
                    'y_pred': y_pred,
                    'y_proba': y_proba
                }
                
                self.logger.info(f"{ensemble_name} - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}")
                
            except Exception as e:
                self.logger.error(f"Failed to evaluate {ensemble_name}: {e}")
                continue
        
        # Create model comparison
        if self.results['test_results']:
            comparison_df = compare_models(self.results['test_results'])
            self.results['model_comparison'] = comparison_df
            
            # Log best performing model
            best_model = comparison_df.loc[comparison_df['f1'].idxmax(), 'model']
            best_f1 = comparison_df.loc[comparison_df['f1'].idxmax(), 'f1']
            self.logger.info(f"Best performing model: {best_model} (F1: {best_f1:.4f})")
        
        self.logger.info("Model evaluation completed")
    
    def analyze_ensemble_decisions(self) -> None:
        """Analyze where ensembles improve over base models."""
        self.logger.info("Analyzing ensemble decisions...")
        
        try:
            if not self.ensemble_models or not self.base_models:
                self.logger.warning("Need both base and ensemble models for decision analysis")
                return
            
            # Compare ensemble vs base model predictions
            analysis_results = {}
            
            for ensemble_name, ensemble_model in self.ensemble_models.items():
                ensemble_preds = ensemble_model.predict(self.X_test)
                
                # Compare with each base model
                for base_name, base_model in self.base_models.items():
                    base_preds = base_model.predict(self.X_test)
                    
                    # Find cases where ensemble and base model disagree
                    disagreements = ensemble_preds != base_preds
                    n_disagreements = np.sum(disagreements)
                    
                    if n_disagreements > 0:
                        # Analyze where ensemble is correct vs base model
                        ensemble_correct = (ensemble_preds == self.y_test) & disagreements
                        base_correct = (base_preds == self.y_test) & disagreements
                        
                        ensemble_improvements = np.sum(ensemble_correct & ~base_correct)
                        ensemble_degradations = np.sum(~ensemble_correct & base_correct)
                        
                        analysis_results[f"{ensemble_name}_vs_{base_name}"] = {
                            'total_disagreements': int(n_disagreements),
                            'disagreement_rate': float(n_disagreements / len(self.y_test)),
                            'ensemble_improvements': int(ensemble_improvements),
                            'ensemble_degradations': int(ensemble_degradations),
                            'net_improvement': int(ensemble_improvements - ensemble_degradations)
                        }
            
            self.results['ensemble_analysis'] = analysis_results
            
            # Log key findings
            for comparison, stats in analysis_results.items():
                self.logger.info(
                    f"{comparison}: {stats['total_disagreements']} disagreements, "
                    f"net improvement: {stats['net_improvement']} samples"
                )
            
        except Exception as e:
            self.logger.warning(f"Error in ensemble decision analysis: {e}")
    
    def analyze_results(self) -> None:
        """Analyze and interpret ensemble results."""
        super().analyze_results()
        
        # Additional ensemble-specific analysis
        self.analyze_ensemble_decisions()
        
        # Create visualizations
        self.logger.info("Creating ensemble visualizations...")
        
        try:
            plots_dir = os.path.join(self.output_dir, "plots")
            plot_results(self.results, plots_dir)
            
            # Create ensemble-specific plots
            self._create_ensemble_plots(plots_dir)
            
            self.logger.info("Visualizations created successfully")
            
        except Exception as e:
            self.logger.warning(f"Error creating visualizations: {e}")
    
    def _create_ensemble_plots(self, plots_dir: str) -> None:
        """Create ensemble-specific visualizations."""
        try:
            from ..evaluation.visualizer import plot_model_comparison
            
            if 'model_comparison' in self.results:
                plot_model_comparison(
                    self.results['model_comparison'],
                    save_path=os.path.join(plots_dir, "ensemble_comparison.png")
                )
            
            # Create ensemble decision analysis plots
            if 'ensemble_analysis' in self.results:
                import matplotlib.pyplot as plt
                
                analysis_data = self.results['ensemble_analysis']
                
                fig, ax = plt.subplots(figsize=(12, 6))
                
                comparisons = list(analysis_data.keys())
                improvements = [analysis_data[comp]['net_improvement'] for comp in comparisons]
                
                bars = ax.bar(range(len(comparisons)), improvements)
                ax.set_xticks(range(len(comparisons)))
                ax.set_xticklabels(comparisons, rotation=45, ha='right')
                ax.set_ylabel('Net Improvement (samples)')
                ax.set_title('Ensemble vs Base Model Performance')
                ax.grid(True, alpha=0.3)
                
                # Color bars based on improvement/degradation
                for bar, improvement in zip(bars, improvements):
                    bar.set_color('green' if improvement > 0 else 'red')
                
                plt.tight_layout()
                plt.savefig(os.path.join(plots_dir, "ensemble_improvements.png"), dpi=300)
                plt.close()
            
        except Exception as e:
            self.logger.warning(f"Error creating ensemble-specific plots: {e}")
    
    def save_ensemble_weights(self) -> None:
        """Save ensemble weights and configurations."""
        try:
            ensemble_info = {}
            
            for ensemble_name, ensemble_model in self.ensemble_models.items():
                info = {'type': ensemble_name}
                
                if hasattr(ensemble_model, 'weights') and ensemble_model.weights is not None:
                    info['weights'] = ensemble_model.weights.tolist()
                
                if hasattr(ensemble_model, 'meta_learner'):
                    info['meta_learner_type'] = ensemble_model.meta_learner_type
                
                if hasattr(ensemble_model, 'voting_strategy'):
                    info['voting_strategy'] = ensemble_model.voting_strategy
                
                ensemble_info[ensemble_name] = info
            
            # Save to JSON
            import json
            ensemble_path = os.path.join(self.output_dir, "results", "ensemble_configurations.json")
            with open(ensemble_path, 'w') as f:
                json.dump(ensemble_info, f, indent=2)
            
            self.logger.info(f"Ensemble configurations saved to {ensemble_path}")
            
        except Exception as e:
            self.logger.warning(f"Error saving ensemble weights: {e}")
    
    def save_artifacts(self) -> None:
        """Save all artifacts including ensemble-specific information."""
        # Save ensemble weights and configurations
        self.save_ensemble_weights()
        
        # Call parent method
        super().save_artifacts()