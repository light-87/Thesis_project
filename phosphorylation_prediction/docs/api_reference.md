# API Reference

This document provides detailed API documentation for the Phosphorylation Site Prediction Framework.

## Configuration Module (`config`)

### config.load_config()

```python
def load_config(config_path: str) -> Dict[str, Any]
```

Load and validate configuration from YAML file.

**Parameters:**
- `config_path` (str): Path to YAML configuration file

**Returns:**
- `Dict[str, Any]`: Loaded and validated configuration

**Example:**
```python
from config import load_config

config = load_config("config/default_config.yaml")
print(config['model']['type'])  # 'xgboost'
```

### config.save_config()

```python
def save_config(config: Dict[str, Any], config_path: str) -> None
```

Save configuration to YAML file.

**Parameters:**
- `config` (Dict[str, Any]): Configuration dictionary
- `config_path` (str): Output file path

### config.validate_config()

```python
def validate_config(config: Dict[str, Any]) -> Tuple[bool, List[str]]
```

Validate configuration structure and values.

**Parameters:**
- `config` (Dict[str, Any]): Configuration to validate

**Returns:**
- `Tuple[bool, List[str]]`: (is_valid, error_messages)

## Data Module (`data`)

### SequenceDataset

```python
class SequenceDataset:
    def __init__(self, sequences: List[str], labels: Optional[List[int]] = None, 
                 metadata: Optional[Dict[str, Any]] = None)
```

Dataset class for protein sequences.

**Parameters:**
- `sequences` (List[str]): List of protein sequences
- `labels` (Optional[List[int]]): Binary labels (0/1)
- `metadata` (Optional[Dict[str, Any]]): Additional metadata

**Methods:**

#### from_csv()
```python
@classmethod
def from_csv(cls, file_path: str) -> 'SequenceDataset'
```

Load dataset from CSV file.

#### split()
```python
def split(self, ratios: Dict[str, float], strategy: str = "random") -> Tuple['SequenceDataset', ...]
```

Split dataset into train/validation/test sets.

**Parameters:**
- `ratios` (Dict[str, float]): Split ratios (e.g., {'train': 0.7, 'val': 0.15, 'test': 0.15})
- `strategy` (str): Split strategy ('random' or 'protein_level')

#### balance_classes()
```python
def balance_classes(self, method: str = "undersample") -> 'SequenceDataset'
```

Balance class distribution.

### SequenceProcessor

```python
class SequenceProcessor:
    def __init__(self, window_size: int = 21, tokenizer_name: Optional[str] = None)
```

Process protein sequences for model input.

**Parameters:**
- `window_size` (int): Size of sequence windows
- `tokenizer_name` (Optional[str]): Tokenizer for transformer models

**Methods:**

#### extract_windows()
```python
def extract_windows(self, sequence: str, positions: Optional[List[int]] = None) -> Tuple[List[str], List[int]]
```

Extract sequence windows around positions.

#### tokenize_sequences()
```python
def tokenize_sequences(self, sequences: List[str]) -> Dict[str, torch.Tensor]
```

Tokenize sequences for transformer models.

### FeatureExtractor

```python
class FeatureExtractor:
    def __init__(self, features_config: Optional[Dict[str, bool]] = None)
```

Extract features from protein sequences.

**Methods:**

#### extract_features()
```python
def extract_features(self, sequences: List[str]) -> np.ndarray
```

Extract comprehensive features from sequences.

#### get_feature_names()
```python
def get_feature_names(self) -> List[str]
```

Get names of extracted features.

#### extract_aac()
```python
def extract_aac(self, sequences: List[str]) -> np.ndarray
```

Extract amino acid composition features.

#### extract_dpc()
```python
def extract_dpc(self, sequences: List[str]) -> np.ndarray
```

Extract dipeptide composition features.

## Models Module (`models`)

### BaseModel

```python
class BaseModel(ABC):
    def __init__(self, config: Dict[str, Any])
```

Abstract base class for all models.

**Methods:**

#### fit()
```python
@abstractmethod
def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs) -> None
```

Train the model.

#### predict()
```python
@abstractmethod
def predict(self, X) -> np.ndarray
```

Make binary predictions.

#### predict_proba()
```python
@abstractmethod
def predict_proba(self, X) -> np.ndarray
```

Predict class probabilities.

### XGBoostModel

```python
class XGBoostModel(BaseModel):
    def __init__(self, config: Dict[str, Any])
```

XGBoost gradient boosting model.

**Methods:**

#### get_feature_importance()
```python
def get_feature_importance(self) -> np.ndarray
```

Get feature importance scores.

#### save_model()
```python
def save_model(self, file_path: str) -> None
```

Save trained model.

#### load_model()
```python
def load_model(self, file_path: str) -> None
```

Load pre-trained model.

### TransformerModel

```python
class TransformerModel(BaseModel):
    def __init__(self, config: Dict[str, Any])
```

Transformer model using ESM2 backbone.

**Methods:**

#### get_attention_weights()
```python
def get_attention_weights(self, sequences: List[str]) -> np.ndarray
```

Extract attention weights for sequences.

#### freeze_backbone()
```python
def freeze_backbone(self, freeze: bool = True) -> None
```

Freeze/unfreeze ESM2 backbone.

### Ensemble Models

#### VotingEnsemble

```python
class VotingEnsemble(BaseModel):
    def __init__(self, config: Dict[str, Any])
```

Ensemble using weighted voting.

#### StackingEnsemble

```python
class StackingEnsemble(BaseModel):
    def __init__(self, config: Dict[str, Any])
```

Ensemble using stacking with meta-learner.

#### DynamicEnsemble

```python
class DynamicEnsemble(BaseModel):
    def __init__(self, config: Dict[str, Any])
```

Dynamic ensemble with competence-based selection.

## Training Module (`training`)

### Trainer

```python
class Trainer:
    def __init__(self, config: Dict[str, Any])
```

Model training orchestrator.

**Methods:**

#### fit()
```python
def fit(self, model: BaseModel, train_data, val_data=None) -> Dict[str, Any]
```

Train model with callbacks and monitoring.

#### add_callback()
```python
def add_callback(self, callback: Callback) -> None
```

Add training callback.

### Callbacks

#### EarlyStopping

```python
class EarlyStopping(Callback):
    def __init__(self, monitor: str = "val_loss", patience: int = 10, 
                 mode: str = "min", min_delta: float = 0.0)
```

Early stopping callback.

#### ModelCheckpoint

```python
class ModelCheckpoint(Callback):
    def __init__(self, filepath: str, monitor: str = "val_loss", 
                 save_best_only: bool = True, mode: str = "min")
```

Model checkpointing callback.

#### WandbCallback

```python
class WandbCallback(Callback):
    def __init__(self, project: str, run_name: Optional[str] = None)
```

Weights & Biases logging callback.

## Evaluation Module (`evaluation`)

### ModelEvaluator

```python
class ModelEvaluator:
    def __init__(self, config: Dict[str, Any])
```

Comprehensive model evaluation.

**Methods:**

#### evaluate()
```python
def evaluate(self, model: BaseModel, test_data) -> Dict[str, Any]
```

Evaluate model on test data.

#### calculate_metrics()
```python
def calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     y_prob: Optional[np.ndarray] = None) -> Dict[str, float]
```

Calculate evaluation metrics.

#### bootstrap_metrics()
```python
def bootstrap_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, 
                     n_samples: int = 1000) -> Dict[str, Dict[str, float]]
```

Calculate bootstrap confidence intervals.

### Visualization Functions

#### plot_confusion_matrix()
```python
def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, 
                         save_path: Optional[str] = None) -> None
```

Plot confusion matrix.

#### plot_roc_curve()
```python
def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray, 
                  save_path: Optional[str] = None) -> None
```

Plot ROC curve.

#### plot_feature_importance()
```python
def plot_feature_importance(model, feature_names: Optional[List[str]] = None,
                           top_k: int = 20, save_path: Optional[str] = None) -> None
```

Plot feature importance.

### ModelAnalyzer

```python
class ModelAnalyzer:
    def __init__(self)
```

Advanced model analysis tools.

**Methods:**

#### analyze_errors()
```python
def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                  X: Optional[np.ndarray] = None) -> Dict[str, Any]
```

Analyze prediction errors.

#### compare_models()
```python
def compare_models(self, models: Dict[str, BaseModel], test_data) -> Dict[str, Any]
```

Compare multiple models.

## Experiments Module (`experiments`)

### BaseExperiment

```python
class BaseExperiment(ABC):
    def __init__(self, config: Dict[str, Any], output_dir: str)
```

Base class for experiments.

**Methods:**

#### run()
```python
def run(self) -> Dict[str, Any]
```

Run complete experiment.

#### load_data()
```python
@abstractmethod
def load_data(self, data_path: str) -> Any
```

Load experiment data.

### SingleModelExperiment

```python
class SingleModelExperiment(BaseExperiment):
    def __init__(self, config: Dict[str, Any], output_dir: str, model_type: str)
```

Single model training experiment.

### EnsembleExperiment

```python
class EnsembleExperiment(BaseExperiment):
    def __init__(self, config: Dict[str, Any], output_dir: str)
```

Ensemble model experiment.

### CrossValidationExperiment

```python
class CrossValidationExperiment(BaseExperiment):
    def __init__(self, config: Dict[str, Any], output_dir: str, 
                 model_type: str, n_folds: int = 5)
```

Cross-validation experiment.

## Utils Module (`utils`)

### Logging

#### get_logger()
```python
def get_logger(name: str, level: str = "INFO") -> logging.Logger
```

Get configured logger.

#### ExperimentLogger

```python
class ExperimentLogger:
    def __init__(self, experiment_name: str, log_dir: str)
```

Experiment-specific logger.

### Reproducibility

#### set_seed()
```python
def set_seed(seed: int) -> None
```

Set random seeds for reproducibility.

#### ReproducibleRandom

```python
class ReproducibleRandom:
    def __init__(self, seed: int)
```

Context manager for reproducible random operations.

### Memory Management

#### MemoryManager

```python
class MemoryManager:
    def __init__(self, gpu_id: Optional[int] = None)
```

Memory monitoring and management.

**Methods:**

#### get_memory_info()
```python
def get_memory_info(self) -> Dict[str, float]
```

Get current memory usage.

#### estimate_batch_size()
```python
def estimate_batch_size(self, model_memory_gb: float, 
                       sample_size_mb: float) -> int
```

Estimate optimal batch size.

#### MemoryProfiler

```python
class MemoryProfiler:
    def __init__(self, name: str = "Operation")
```

Context manager for memory profiling.

### Checkpointing

#### CheckpointManager

```python
class CheckpointManager:
    def __init__(self, checkpoint_dir: str, max_checkpoints: int = 5)
```

Manage model checkpoints.

**Methods:**

#### save_checkpoint()
```python
def save_checkpoint(self, checkpoint: Checkpoint, is_best: bool = False) -> str
```

Save model checkpoint.

#### load_checkpoint()
```python
def load_checkpoint(self, filename: Optional[str] = None, 
                   load_best: bool = False) -> Optional[Checkpoint]
```

Load model checkpoint.

## Error Handling

### Common Exceptions

#### ConfigurationError
```python
class ConfigurationError(Exception):
    """Raised when configuration is invalid."""
    pass
```

#### ModelError
```python
class ModelError(Exception):
    """Raised when model operations fail."""
    pass
```

#### DataError
```python
class DataError(Exception):
    """Raised when data processing fails."""
    pass
```

## Type Hints

The framework uses comprehensive type hints for better IDE support and code documentation:

```python
from typing import List, Dict, Optional, Tuple, Union, Any
import numpy as np
import torch

# Common type aliases
SequenceList = List[str]
LabelList = List[int]
ConfigDict = Dict[str, Any]
MetricsDict = Dict[str, float]
```

## Example Usage Patterns

### Basic Training Pipeline

```python
from config import load_config
from data import SequenceDataset, SequenceProcessor, FeatureExtractor
from models import XGBoostModel
from training import Trainer
from evaluation import ModelEvaluator

# Load configuration
config = load_config("config.yaml")

# Prepare data
dataset = SequenceDataset.from_csv("data.csv")
train_data, val_data, test_data = dataset.split(config['data'])

# Process sequences
processor = SequenceProcessor(config['data'])
extractor = FeatureExtractor(config['model']['features'])

# Extract features
X_train = extractor.extract_features(train_data.sequences)
y_train = np.array(train_data.labels)

# Train model
model = XGBoostModel(config['model'])
trainer = Trainer(config['training'])
trainer.fit(model, (X_train, y_train))

# Evaluate
evaluator = ModelEvaluator(config['evaluation'])
results = evaluator.evaluate(model, test_data)
```

### Ensemble Pipeline

```python
from models import VotingEnsemble

# Configure ensemble
ensemble_config = {
    'type': 'ensemble',
    'ensemble_type': 'voting',
    'base_models': [
        {'type': 'xgboost', 'weight': 1.0},
        {'type': 'transformer', 'weight': 1.5}
    ]
}

# Train ensemble
ensemble = VotingEnsemble(ensemble_config)
ensemble.fit(train_data, val_data)

# Evaluate
results = evaluator.evaluate(ensemble, test_data)
```

### Cross-Validation

```python
from experiments import CrossValidationExperiment

# Run cross-validation experiment
cv_experiment = CrossValidationExperiment(
    config=config,
    output_dir="cv_results",
    model_type="xgboost",
    n_folds=5
)

cv_experiment.load_data("data.csv")
cv_results = cv_experiment.run()
```

This API reference provides comprehensive documentation for all major classes and functions in the framework, including parameters, return types, and usage examples.