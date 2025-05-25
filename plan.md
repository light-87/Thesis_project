# plan.md - Phosphorylation Site Prediction: Complete Implementation Plan

## Project Context and Background

### What is Phosphorylation Site Prediction?
Phosphorylation is a biological process where a phosphate group is added to specific amino acids (Serine/S, Threonine/T, or Tyrosine/Y) in proteins. This modification changes how proteins function and is crucial for cellular signaling. Our task is to predict which S/T/Y sites in a protein sequence will be phosphorylated.

### Current State
- **Data**: Protein sequences with known phosphorylation sites from UniProt database
- **Positive samples**: Known phosphorylated S/T/Y sites
- **Negative samples**: Randomly selected non-phosphorylated S/T/Y sites (balanced 50/50)
- **Current models**: 
  - XGBoost with engineered features (82.8% accuracy)
  - Transformer using ESM2 protein language model (80.22% accuracy)
  - Simple ensemble methods achieving ~82.8% accuracy

### Problem
The current codebase is scattered across multiple notebooks and scripts without proper structure, making it difficult to:
- Run systematic experiments
- Track results properly
- Implement advanced ensemble methods
- Maintain and extend the code

## Project Goals

### Primary Objectives
1. **Standardize the codebase** into a modular, maintainable structure
2. **Implement comprehensive experiment tracking** using Weights & Biases
3. **Create robust cross-validation pipeline** for reliable evaluation
4. **Develop advanced ensemble methods** to improve prediction accuracy
5. **Build interpretability tools** for model analysis
6. **Optimize training speed** especially for transformer models

### Technical Requirements
- Python 3.8+
- PyTorch for deep learning models
- XGBoost for gradient boosting
- Transformers library for protein language models
- Weights & Biases for experiment tracking
- Support for GPU acceleration
- Memory-efficient processing for large datasets

## Detailed File Structure and Implementation

### Project Root Structure
```
phosphorylation_prediction/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example                  # Example environment variables
├── config/
├── src/
├── scripts/
├── notebooks/
├── tests/
├── data/                        # Data directory (git-ignored)
├── outputs/                     # Results directory (git-ignored)
└── docs/
```

### Detailed Implementation Specifications

#### **1. Configuration Module (`config/`)**

**`config/__init__.py`**
```python
# Exports configuration loading utilities
from .config_loader import load_config, save_config
from .model_configs import XGBoostConfig, TransformerConfig, EnsembleConfig
```

**`config/config_loader.py`**
Functions:
- `load_config(config_path: str) -> Dict`: Load YAML config and merge with defaults
- `save_config(config: Dict, path: str)`: Save configuration for reproducibility
- `merge_configs(base: Dict, override: Dict) -> Dict`: Merge configurations
- `validate_config(config: Dict)`: Validate configuration parameters

**`config/model_configs.py`**
Classes:
- `BaseConfig`: Base configuration dataclass with common parameters
- `XGBoostConfig`: XGBoost-specific hyperparameters
- `TransformerConfig`: Transformer model configuration
- `EnsembleConfig`: Ensemble method configurations
- `DataConfig`: Data processing parameters
- `TrainingConfig`: Training loop parameters
- `WandbConfig`: Weights & Biases settings

**`config/default_config.yaml`**
```yaml
experiment:
  project_name: "phosphorylation_prediction"
  run_name: null  # Auto-generated if null
  seed: 42
  device: "cuda"
  num_workers: 4
  
wandb:
  entity: "your-entity"
  project: "phospho-prediction"
  tags: ["baseline"]
  log_frequency: 50
  log_predictions: true
  
data:
  sequence_col: "Sequence"
  position_col: "Position"
  target_col: "target"
  protein_id_col: "Header"
  window_size: 10
  max_sequence_length: 5000
  
preprocessing:
  balance_classes: true
  augmentation: false
  
features:
  use_aac: true
  use_dpc: true
  use_tpc: true
  use_binary: true
  use_physicochemical: true
  
cross_validation:
  n_folds: 5
  strategy: "stratified_group"  # Stratified by target, grouped by protein
  
xgboost:
  n_estimators: 1000
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  early_stopping_rounds: 50
  eval_metric: ["logloss", "auc"]
  
transformer:
  model_name: "facebook/esm2_t6_8M_UR50D"
  max_length: 64
  batch_size: 32
  learning_rate: 2e-5
  num_epochs: 10
  warmup_steps: 500
  gradient_accumulation_steps: 4
  fp16: true
  
ensemble:
  methods:
    voting:
      strategy: "soft"  # soft or hard
      weights: "optimize"  # optimize, equal, or list
    stacking:
      meta_learner: "logistic_regression"
      use_probas: true
      cv_predictions: true
    blending:
      blend_ratio: 0.2
    dynamic:
      k_neighbors: 5
      similarity_metric: "cosine"
```

#### **2. Data Module (`src/data/`)**

**`src/data/__init__.py`**
```python
from .dataset import PhosphoPredictionDataset, FeatureDataset
from .data_loader import create_data_loader, create_cv_splits
from .preprocessor import FeatureExtractor, SequenceProcessor
```

**`src/data/dataset.py`**
Classes:
- `PhosphoPredictionDataset(Dataset)`: Base dataset class for protein sequences
  - `__init__(self, data_path, config)`: Initialize with data and configuration
  - `__len__()`: Return dataset size
  - `__getitem__(idx)`: Return processed sample
  - `get_protein_groups()`: Return protein groupings for CV

- `FeatureDataset(Dataset)`: Dataset for pre-extracted features
  - `__init__(self, features, labels, config)`: Initialize with features
  - `normalize_features()`: Normalize feature values
  - `get_feature_names()`: Return feature column names

**`src/data/data_loader.py`**
Functions:
- `create_data_loader(dataset, config, is_training=True) -> DataLoader`
- `create_cv_splits(data, config) -> List[Tuple[np.ndarray, np.ndarray]]`
- `stratified_group_split(X, y, groups, n_splits) -> Iterator`
- `load_raw_data(data_path) -> pd.DataFrame`
- `generate_negative_samples(df, random_state) -> pd.DataFrame`

**`src/data/preprocessor.py`**
Classes:
- `SequenceProcessor`:
  - `extract_window(sequence, position, window_size) -> str`
  - `tokenize_sequence(sequence, tokenizer) -> Dict`
  - `pad_sequence(sequence, max_length) -> str`
  
- `FeatureExtractor`:
  - `extract_aac(sequence) -> Dict[str, float]`
  - `extract_dpc(sequence) -> Dict[str, float]`
  - `extract_tpc(sequence) -> Dict[str, float]`
  - `extract_binary_encoding(sequence, position, window_size) -> np.ndarray`
  - `extract_physicochemical(sequence, position, window_size) -> np.ndarray`
  - `extract_all_features(sequence, position, config) -> np.ndarray`

#### **3. Models Module (`src/models/`)**

**`src/models/__init__.py`**
```python
from .base_model import BaseModel
from .xgboost_model import XGBoostModel
from .transformer_model import TransformerModel, PhosTransformer
from .ensemble_models import VotingEnsemble, StackingEnsemble, DynamicEnsemble
```

**`src/models/base_model.py`**
```python
class BaseModel(ABC):
    """Abstract base class for all models"""
    
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger
        self.model = None
    
    @abstractmethod
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train the model"""
        pass
    
    @abstractmethod
    def predict(self, X) -> np.ndarray:
        """Return class predictions"""
        pass
    
    @abstractmethod
    def predict_proba(self, X) -> np.ndarray:
        """Return probability predictions"""
        pass
    
    @abstractmethod
    def save(self, path: str):
        """Save model to disk"""
        pass
    
    @abstractmethod
    def load(self, path: str):
        """Load model from disk"""
        pass
    
    def get_params(self) -> Dict:
        """Get model parameters"""
        return self.config
```

**`src/models/xgboost_model.py`**
```python
class XGBoostModel(BaseModel):
    """XGBoost wrapper implementing BaseModel interface"""
    
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.feature_importance_ = None
        
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        """Train XGBoost with early stopping"""
        # Implementation
        
    def predict_proba(self, X) -> np.ndarray:
        """Get probability predictions"""
        # Implementation
        
    def get_feature_importance(self, importance_type='gain') -> Dict:
        """Get feature importance scores"""
        # Implementation
```

**`src/models/transformer_model.py`**
```python
class PhosTransformer(nn.Module):
    """Transformer architecture for phosphorylation prediction"""
    
    def __init__(self, config):
        super().__init__()
        self.encoder = AutoModel.from_pretrained(config.model_name)
        self.dropout = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.hidden_size, 1)
        
    def forward(self, input_ids, attention_mask):
        """Forward pass"""
        # Implementation with context aggregation

class TransformerModel(BaseModel):
    """Transformer wrapper implementing BaseModel interface"""
    
    def __init__(self, config, logger=None):
        super().__init__(config, logger)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model_name)
        self.device = torch.device(config.device)
        
    def fit(self, sequences, positions, labels, val_data=None):
        """Train transformer with mixed precision"""
        # Implementation with wandb logging
        
    def get_attention_weights(self, sequence, position):
        """Extract attention weights for interpretability"""
        # Implementation
```

**`src/models/ensemble_models.py`**
```python
class VotingEnsemble(BaseModel):
    """Voting ensemble with optimized weights"""
    
    def __init__(self, models, config, logger=None):
        super().__init__(config, logger)
        self.models = models
        self.weights = None
        
    def optimize_weights(self, X_val, y_val, metric='f1'):
        """Find optimal voting weights"""
        # Implementation
        
class StackingEnsemble(BaseModel):
    """Stacking ensemble with meta-learner"""
    
    def __init__(self, models, meta_learner, config, logger=None):
        super().__init__(config, logger)
        self.models = models
        self.meta_learner = meta_learner
        
    def create_meta_features(self, X):
        """Generate meta-features from base models"""
        # Implementation

class DynamicEnsemble(BaseModel):
    """Dynamic model selection based on input similarity"""
    
    def __init__(self, models, config, logger=None):
        super().__init__(config, logger)
        self.models = models
        self.competence_regions = None
        
    def estimate_competence(self, X):
        """Estimate model competence for each input"""
        # Implementation
```

#### **4. Training Module (`src/training/`)**

**`src/training/__init__.py`**
```python
from .trainer import BaseTrainer, create_trainer
from .callbacks import EarlyStopping, ModelCheckpoint, WandbCallback
```

**`src/training/trainer.py`**
```python
class BaseTrainer:
    """Base trainer class with wandb integration"""
    
    def __init__(self, model, config, logger):
        self.model = model
        self.config = config
        self.logger = logger
        self.callbacks = []
        
    def add_callback(self, callback):
        """Add training callback"""
        self.callbacks.append(callback)
        
    def train(self, train_data, val_data=None):
        """Main training loop"""
        wandb.init(
            project=self.config.wandb.project,
            config=self.config,
            tags=self.config.wandb.tags
        )
        # Implementation with callback hooks
        
    def evaluate(self, test_data):
        """Evaluate model and log to wandb"""
        # Implementation

def create_trainer(model_type, model, config, logger):
    """Factory function for creating appropriate trainer"""
    # Return XGBoostTrainer, TransformerTrainer, or EnsembleTrainer
```

**`src/training/callbacks.py`**
```python
class Callback:
    """Base callback class"""
    def on_epoch_start(self, epoch, logs=None): pass
    def on_epoch_end(self, epoch, logs=None): pass
    def on_batch_start(self, batch, logs=None): pass
    def on_batch_end(self, batch, logs=None): pass
    
class WandbCallback(Callback):
    """Weights & Biases logging callback"""
    def __init__(self, log_frequency=50):
        self.log_frequency = log_frequency
        
    def on_batch_end(self, batch, logs=None):
        if batch % self.log_frequency == 0:
            wandb.log(logs)
            
class EarlyStopping(Callback):
    """Early stopping with patience"""
    def __init__(self, patience=5, metric='val_loss', mode='min'):
        self.patience = patience
        self.metric = metric
        self.mode = mode
        self.best_score = None
        self.counter = 0
```

#### **5. Evaluation Module (`src/evaluation/`)**

**`src/evaluation/__init__.py`**
```python
from .metrics import calculate_metrics, calculate_cv_metrics
from .visualizer import plot_results, create_dashboard
from .analyzer import ErrorAnalyzer, InterpretabilityAnalyzer
```

**`src/evaluation/metrics.py`**
Functions:
- `calculate_metrics(y_true, y_pred, y_proba=None) -> Dict`
- `calculate_cv_metrics(cv_results) -> Dict`
- `calculate_per_class_metrics(y_true, y_pred) -> Dict`
- `calculate_confidence_calibration(y_true, y_proba) -> Dict`

**`src/evaluation/visualizer.py`**
Functions:
- `plot_roc_curves(results_dict, save_path=None)`
- `plot_precision_recall_curves(results_dict, save_path=None)`
- `plot_confusion_matrices(results_dict, save_path=None)`
- `plot_feature_importance(importance_dict, top_n=30, save_path=None)`
- `create_dashboard(results, save_path)`: Create comprehensive HTML dashboard

**`src/evaluation/analyzer.py`**
Classes:
- `ErrorAnalyzer`:
  - `analyze_errors(y_true, y_pred, features) -> Dict`
  - `find_error_patterns(errors, features) -> List[Dict]`
  - `cluster_errors(error_features, n_clusters=5) -> np.ndarray`
  
- `InterpretabilityAnalyzer`:
  - `calculate_shap_values(model, X, sample_size=100) -> np.ndarray`
  - `analyze_attention_patterns(transformer_model, sequences) -> Dict`
  - `compare_model_decisions(models, X) -> pd.DataFrame`

#### **6. Experiments Module (`src/experiments/`)**

**`src/experiments/__init__.py`**
```python
from .base_experiment import BaseExperiment
from .single_model_experiment import SingleModelExperiment
from .ensemble_experiment import EnsembleExperiment
from .cross_validation_experiment import CrossValidationExperiment
```

**`src/experiments/base_experiment.py`**
```python
class BaseExperiment:
    """Template for all experiments"""
    
    def __init__(self, config_path):
        self.config = load_config(config_path)
        self.logger = self.setup_logging()
        self.setup_wandb()
        
    def setup_wandb(self):
        """Initialize wandb run"""
        wandb.init(
            project=self.config.wandb.project,
            entity=self.config.wandb.entity,
            config=self.config,
            name=self.generate_run_name()
        )
        
    def run(self):
        """Main experiment pipeline"""
        self.load_data()
        self.preprocess_data()
        self.train_models()
        self.evaluate_models()
        self.analyze_results()
        self.save_artifacts()
        
    @abstractmethod
    def train_models(self): pass
    
    def save_artifacts(self):
        """Save all artifacts to wandb"""
        # Save models, configs, results
```

**`src/experiments/ensemble_experiment.py`**
```python
class EnsembleExperiment(BaseExperiment):
    """Experiment for ensemble methods"""
    
    def train_models(self):
        # Train base models
        self.train_base_models()
        
        # Train ensemble methods
        self.train_voting_ensemble()
        self.train_stacking_ensemble()
        self.train_dynamic_ensemble()
        
    def train_voting_ensemble(self):
        """Train voting ensemble with weight optimization"""
        # Implementation
        
    def analyze_ensemble_decisions(self):
        """Analyze where ensembles improve over base models"""
        # Implementation
```

#### **7. Utilities Module (`src/utils/`)**

**`src/utils/__init__.py`**
```python
from .logger import setup_logger, get_logger
from .checkpoint import save_checkpoint, load_checkpoint
from .reproducibility import set_seed, ensure_reproducibility
from .memory import optimize_memory, get_memory_usage
```

**`src/utils/logger.py`**
Functions:
- `setup_logger(name, log_file, level=logging.INFO) -> logging.Logger`
- `get_logger(name) -> logging.Logger`

**`src/utils/checkpoint.py`**
Functions:
- `save_checkpoint(state, filepath, is_best=False)`
- `load_checkpoint(filepath, model, optimizer=None)`
- `save_experiment_state(experiment, filepath)`

**`src/utils/reproducibility.py`**
Functions:
- `set_seed(seed: int)`
- `ensure_reproducibility(seed: int)`: Set all random seeds
- `get_random_state(seed: int) -> np.random.RandomState`

**`src/utils/memory.py`**
Functions:
- `optimize_memory()`: Clear GPU cache and run garbage collection
- `get_memory_usage() -> Dict`: Get current CPU and GPU memory
- `estimate_batch_size(model, input_shape) -> int`: Estimate optimal batch size

#### **8. Scripts (`scripts/`)**

**`scripts/train_single_model.py`**
```python
"""Script to train a single model (XGBoost or Transformer)"""
import argparse
from src.experiments import SingleModelExperiment

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['xgboost', 'transformer'])
    parser.add_argument('--config', default='config/default_config.yaml')
    parser.add_argument('--data', required=True)
    args = parser.parse_args()
    
    experiment = SingleModelExperiment(args.config)
    experiment.run()

if __name__ == '__main__':
    main()
```

**`scripts/train_ensemble.py`**
```python
"""Script to train ensemble models"""
# Similar structure for ensemble training
```

**`scripts/run_cross_validation.py`**
```python
"""Script to run cross-validation experiments"""
# Implementation for CV experiments
```

**`scripts/analyze_results.py`**
```python
"""Script to analyze and visualize results"""
# Load results from wandb and create visualizations
```

#### **9. Main Entry Point (`main.py`)**
```python
"""Main entry point for the phosphorylation prediction project"""
import argparse
import os
from src.experiments import (
    SingleModelExperiment,
    EnsembleExperiment,
    CrossValidationExperiment
)

def main():
    parser = argparse.ArgumentParser(
        description='Phosphorylation Site Prediction'
    )
    parser.add_argument(
        '--experiment',
        choices=['single', 'ensemble', 'cv', 'full'],
        default='full',
        help='Type of experiment to run'
    )
    parser.add_argument(
        '--config',
        default='config/default_config.yaml',
        help='Path to configuration file'
    )
    parser.add_argument(
        '--resume',
        help='Path to checkpoint to resume from'
    )
    
    args = parser.parse_args()
    
    # Select experiment type
    if args.experiment == 'single':
        experiment = SingleModelExperiment(args.config)
    elif args.experiment == 'ensemble':
        experiment = EnsembleExperiment(args.config)
    elif args.experiment == 'cv':
        experiment = CrossValidationExperiment(args.config)
    else:
        # Run full pipeline
        experiment = FullPipeline(args.config)
    
    # Run experiment
    experiment.run()

if __name__ == '__main__':
    main()
```

## Implementation Priorities

### Phase 1: Core Infrastructure (Week 1)
1. Set up project structure
2. Implement configuration management
3. Create base model interfaces
4. Set up logging and Weights & Biases integration

### Phase 2: Data Pipeline (Week 1-2)
1. Implement data loading and preprocessing
2. Create feature extraction pipeline
3. Implement cross-validation splits
4. Add data validation and checks

### Phase 3: Model Implementation (Week 2-3)
1. Wrap existing XGBoost model
2. Wrap existing Transformer model
3. Implement training loops with callbacks
4. Add model checkpointing

### Phase 4: Ensemble Methods (Week 3-4)
1. Implement voting ensemble
2. Implement stacking ensemble
3. Implement dynamic selection
4. Add ensemble analysis tools

### Phase 5: Evaluation and Analysis (Week 4)
1. Implement comprehensive metrics
2. Create visualization tools
3. Add interpretability analysis
4. Build results dashboard

## Weights & Biases Integration

### What to Track
1. **Hyperparameters**: All model and training configurations
2. **Metrics**: Loss, accuracy, precision, recall, F1, AUC per epoch
3. **System Metrics**: GPU usage, training time, memory consumption
4. **Artifacts**: Best models, feature importance, predictions
5. **Visualizations**: ROC curves, confusion matrices, learning curves
6. **Custom Metrics**: Per-protein performance, motif-specific accuracy

### Implementation Example
```python
# In trainer
wandb.log({
    'epoch': epoch,
    'train/loss': train_loss,
    'train/accuracy': train_acc,
    'val/loss': val_loss,
    'val/accuracy': val_acc,
    'learning_rate': current_lr,
    'gpu_memory': torch.cuda.memory_allocated(),
})

# Log predictions table
wandb.log({
    'predictions': wandb.Table(
        columns=['sequence', 'position', 'true', 'pred', 'prob'],
        data=prediction_data
    )
})

# Save model artifact
wandb.save('best_model.pt')
```

## Testing Requirements

### Unit Tests (`tests/`)
- Test feature extraction functions
- Test data loading and preprocessing
- Test model interfaces
- Test evaluation metrics

### Integration Tests
- Test full training pipeline
- Test ensemble methods
- Test cross-validation

### Performance Tests
- Memory usage benchmarks
- Training speed benchmarks
- Inference speed tests

## Documentation (`docs/`)
- API documentation
- Usage examples
- Results analysis
- Model architecture diagrams

## Final Notes for Implementation

1. **Error Handling**: Implement comprehensive error handling throughout
2. **Type Hints**: Use type hints for all functions and methods
3. **Docstrings**: Include detailed docstrings for all classes and functions
4. **Progress Bars**: Use tqdm for all long-running operations
5. **Memory Management**: Implement garbage collection after large operations
6. **GPU Support**: Ensure all operations can run on both CPU and GPU
7. **Reproducibility**: Save all random seeds and ensure deterministic behavior
8. **Modularity**: Each component should be independently testable and reusable

This architecture will provide a robust, scalable foundation for phosphorylation site prediction research while maintaining clarity and ease of experimentation.