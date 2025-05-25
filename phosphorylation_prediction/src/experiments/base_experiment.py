"""Base experiment class for phosphorylation prediction."""

import os
import time
import json
import logging
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import load_config
from ..utils.logger import setup_logger
from ..utils.reproducibility import set_seed


class BaseExperiment(ABC):
    """Template for all experiments."""
    
    def __init__(self, config_path: str, experiment_name: Optional[str] = None):
        """
        Initialize base experiment.
        
        Args:
            config_path: Path to configuration file
            experiment_name: Optional experiment name
        """
        self.config_path = config_path
        self.config = load_config(config_path)
        self.experiment_name = experiment_name or self.generate_experiment_name()
        
        # Setup logging
        self.logger = self.setup_logging()
        
        # Setup reproducibility
        self.setup_reproducibility()
        
        # Setup WandB if configured
        self.wandb_run = None
        if self.config.get('wandb', {}).get('enabled', False):
            self.setup_wandb()
        
        # Experiment tracking
        self.start_time = None
        self.end_time = None
        self.results = {}
        self.artifacts = {}
        
        # Create output directories
        self.output_dir = self.create_output_directories()
        
        self.logger.info(f"Initialized experiment: {self.experiment_name}")
        self.logger.info(f"Output directory: {self.output_dir}")
    
    def generate_experiment_name(self) -> str:
        """Generate a unique experiment name."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        project_name = self.config.get('experiment', {}).get('project_name', 'phospho_prediction')
        return f"{project_name}_{timestamp}"
    
    def setup_logging(self) -> logging.Logger:
        """Setup logging for the experiment."""
        log_dir = os.path.join("outputs", self.experiment_name, "logs")
        return setup_logger(
            name=self.experiment_name,
            log_file=os.path.join(log_dir, "experiment.log")
        )
    
    def setup_reproducibility(self) -> None:
        """Setup reproducibility settings."""
        seed = self.config.get('experiment', {}).get('seed', 42)
        set_seed(seed)
        self.logger.info(f"Set random seed to {seed}")
    
    def setup_wandb(self) -> None:
        """Initialize wandb run."""
        try:
            import wandb
            
            wandb_config = self.config.get('wandb', {})
            
            self.wandb_run = wandb.init(
                project=wandb_config.get('project', 'phospho-prediction'),
                entity=wandb_config.get('entity'),
                name=self.experiment_name,
                config=self.config,
                tags=wandb_config.get('tags', []),
                reinit=True
            )
            
            self.logger.info(f"WandB run initialized: {wandb.run.url}")
            
        except ImportError:
            self.logger.warning("WandB not installed, skipping WandB setup")
        except Exception as e:
            self.logger.warning(f"Failed to initialize WandB: {e}")
    
    def create_output_directories(self) -> str:
        """Create output directories for the experiment."""
        base_output_dir = os.path.join("outputs", self.experiment_name)
        
        # Create subdirectories
        subdirs = ["models", "plots", "results", "logs", "artifacts"]
        for subdir in subdirs:
            os.makedirs(os.path.join(base_output_dir, subdir), exist_ok=True)
        
        return base_output_dir
    
    @abstractmethod
    def load_data(self) -> None:
        """Load and prepare data for the experiment."""
        pass
    
    @abstractmethod
    def preprocess_data(self) -> None:
        """Preprocess the loaded data."""
        pass
    
    @abstractmethod
    def train_models(self) -> None:
        """Train the models."""
        pass
    
    @abstractmethod
    def evaluate_models(self) -> None:
        """Evaluate the trained models."""
        pass
    
    def analyze_results(self) -> None:
        """Analyze and interpret results."""
        self.logger.info("Analyzing results...")
        
        try:
            from ..evaluation import ErrorAnalyzer, InterpretabilityAnalyzer
            
            # Basic error analysis
            if 'test_results' in self.results:
                error_analyzer = ErrorAnalyzer(self.logger)
                
                for model_name, model_results in self.results['test_results'].items():
                    if 'y_true' in model_results and 'y_pred' in model_results:
                        error_analysis = error_analyzer.analyze_errors(
                            model_results['y_true'],
                            model_results['y_pred'],
                            model_results.get('features'),
                            model_results.get('additional_data')
                        )
                        
                        self.results[f'{model_name}_error_analysis'] = error_analysis
                        self.logger.info(f"Error analysis completed for {model_name}")
            
            # Interpretability analysis
            if hasattr(self, 'trained_models'):
                interp_analyzer = InterpretabilityAnalyzer(self.logger)
                
                for model_name, model in self.trained_models.items():
                    if hasattr(self, 'X_test') and hasattr(self, 'y_test'):
                        interp_report = interp_analyzer.generate_interpretability_report(
                            model, self.X_test, self.y_test
                        )
                        
                        self.results[f'{model_name}_interpretability'] = interp_report
                        self.logger.info(f"Interpretability analysis completed for {model_name}")
        
        except Exception as e:
            self.logger.warning(f"Error during result analysis: {e}")
    
    def save_artifacts(self) -> None:
        """Save all artifacts to disk and WandB."""
        self.logger.info("Saving artifacts...")
        
        try:
            # Save configuration
            config_path = os.path.join(self.output_dir, "config.json")
            with open(config_path, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            
            # Save results
            results_path = os.path.join(self.output_dir, "results", "results.json")
            with open(results_path, 'w') as f:
                json.dump(self.results, f, indent=2, default=str)
            
            # Save models if available
            if hasattr(self, 'trained_models'):
                for model_name, model in self.trained_models.items():
                    if hasattr(model, 'save'):
                        model_path = os.path.join(self.output_dir, "models", f"{model_name}")
                        model.save(model_path)
                        self.logger.info(f"Saved model: {model_name}")
            
            # Save to WandB if configured
            if self.wandb_run:
                try:
                    import wandb
                    
                    # Log final metrics
                    if 'test_results' in self.results:
                        for model_name, model_results in self.results['test_results'].items():
                            for metric_name, metric_value in model_results.items():
                                if isinstance(metric_value, (int, float)):
                                    wandb.log({f"{model_name}_{metric_name}": metric_value})
                    
                    # Save artifacts
                    wandb.save(config_path)
                    wandb.save(results_path)
                    
                    # Save plots if they exist
                    plots_dir = os.path.join(self.output_dir, "plots")
                    if os.path.exists(plots_dir):
                        for plot_file in os.listdir(plots_dir):
                            if plot_file.endswith('.png'):
                                wandb.save(os.path.join(plots_dir, plot_file))
                    
                except Exception as e:
                    self.logger.warning(f"Error saving to WandB: {e}")
            
            self.logger.info("Artifacts saved successfully")
            
        except Exception as e:
            self.logger.error(f"Error saving artifacts: {e}")
            raise
    
    def run(self) -> Dict[str, Any]:
        """
        Main experiment pipeline.
        
        Returns:
            Experiment results
        """
        self.start_time = time.time()
        self.logger.info(f"Starting experiment: {self.experiment_name}")
        
        try:
            # Run experiment steps
            self.load_data()
            self.preprocess_data()
            self.train_models()
            self.evaluate_models()
            self.analyze_results()
            self.save_artifacts()
            
            self.end_time = time.time()
            duration = self.end_time - self.start_time
            
            self.logger.info(f"Experiment completed successfully in {duration:.2f} seconds")
            
            # Add experiment metadata to results
            self.results['experiment_metadata'] = {
                'experiment_name': self.experiment_name,
                'config_path': self.config_path,
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration_seconds': duration,
                'output_dir': self.output_dir
            }
            
            return self.results
            
        except Exception as e:
            self.end_time = time.time()
            self.logger.error(f"Experiment failed: {e}")
            
            # Save partial results
            try:
                self.save_artifacts()
            except:
                pass
            
            raise
        
        finally:
            # Clean up WandB
            if self.wandb_run:
                try:
                    import wandb
                    wandb.finish()
                except:
                    pass
    
    def get_experiment_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the experiment.
        
        Returns:
            Experiment summary
        """
        summary = {
            'experiment_name': self.experiment_name,
            'config_path': self.config_path,
            'output_dir': self.output_dir,
            'status': 'completed' if self.end_time else 'running' if self.start_time else 'not_started'
        }
        
        if self.start_time:
            summary['start_time'] = datetime.fromtimestamp(self.start_time).isoformat()
        
        if self.end_time:
            summary['end_time'] = datetime.fromtimestamp(self.end_time).isoformat()
            summary['duration_seconds'] = self.end_time - self.start_time
        
        # Add key results if available
        if 'test_results' in self.results:
            summary['test_results'] = {}
            for model_name, model_results in self.results['test_results'].items():
                summary['test_results'][model_name] = {
                    key: value for key, value in model_results.items()
                    if isinstance(value, (int, float)) and not key.startswith('_')
                }
        
        return summary
    
    def cleanup(self) -> None:
        """Clean up experiment resources."""
        if self.wandb_run:
            try:
                import wandb
                wandb.finish()
            except:
                pass
        
        self.logger.info("Experiment cleanup completed")