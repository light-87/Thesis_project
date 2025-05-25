# Configuration Guide

This guide explains how to configure the Phosphorylation Site Prediction Framework for different use cases.

## Configuration File Structure

The framework uses YAML configuration files with the following main sections:

```yaml
# Model configuration
model:
  type: "xgboost"  # Model type
  # Model-specific parameters

# Data processing configuration  
data:
  window_size: 21
  # Data processing parameters

# Training configuration
training:
  epochs: 100
  # Training parameters

# Evaluation configuration
evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score"]
  # Evaluation parameters

# Experiment tracking
wandb:
  project: "phosphorylation-prediction"
  # WandB parameters
```

## Model Configuration

### XGBoost Configuration

```yaml
model:
  type: "xgboost"
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  random_state: 42
  n_jobs: -1
  
  # Feature configuration
  features:
    use_aac: true          # Amino acid composition
    use_dpc: true          # Dipeptide composition  
    use_tpc: false         # Tripeptide composition (memory intensive)
    use_binary: true       # Binary encoding
    use_physicochemical: true  # Physicochemical properties
    
  # Early stopping
  early_stopping_rounds: 10
  eval_metric: "logloss"
```

### Transformer Configuration

```yaml
model:
  type: "transformer"
  
  # Model architecture
  model_name: "facebook/esm2_t6_8M_UR50D"  # ESM2 model variant
  num_labels: 2
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
  
  # Context aggregation
  context_aggregation: "attention"  # Options: mean, max, attention
  freeze_backbone: false  # Whether to freeze ESM2 weights
  
  # Training parameters
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  
  # Mixed precision training
  use_mixed_precision: true
  gradient_accumulation_steps: 1
```

### Ensemble Configuration

```yaml
model:
  type: "ensemble"
  ensemble_type: "voting"  # Options: voting, stacking, dynamic
  
  # Base models to include
  base_models:
    - type: "xgboost"
      weight: 1.0
      config:
        n_estimators: 100
        max_depth: 6
    
    - type: "transformer"
      weight: 1.0
      config:
        model_name: "facebook/esm2_t6_8M_UR50D"
        learning_rate: 2e-5
  
  # Ensemble-specific parameters
  voting_strategy: "soft"  # For voting ensemble
  
  # Stacking ensemble parameters (if ensemble_type: "stacking")
  meta_learner:
    type: "logistic_regression"
    C: 1.0
    
  # Dynamic ensemble parameters (if ensemble_type: "dynamic")
  competence_method: "local_accuracy"
  k_neighbors: 5
```

## Data Configuration

```yaml
data:
  # Window extraction
  window_size: 21
  center_residue: true  # Whether to center on the target residue
  
  # Data splitting
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15
  split_strategy: "protein_level"  # Options: random, protein_level
  
  # Data processing
  tokenizer_name: "facebook/esm2_t6_8M_UR50D"  # For transformer models
  max_sequence_length: 512  # Maximum sequence length
  padding_strategy: "max_length"  # Padding strategy
  
  # Data augmentation (optional)
  augmentation:
    enabled: false
    strategies: ["reverse_complement", "random_crop"]
    augmentation_factor: 2
  
  # Class balancing
  balance_classes: true
  balancing_method: "undersample"  # Options: oversample, undersample, smote
```

## Training Configuration

```yaml
training:
  # General training parameters
  epochs: 100
  batch_size: 32
  validation_batch_size: 64
  
  # Optimization
  optimizer: "adam"  # Options: adam, adamw, sgd
  learning_rate: 0.001
  weight_decay: 0.01
  
  # Learning rate scheduling
  scheduler:
    type: "reduce_on_plateau"  # Options: step, cosine, reduce_on_plateau
    patience: 5
    factor: 0.5
    min_lr: 1e-7
  
  # Early stopping
  early_stopping:
    enabled: true
    patience: 10
    monitor: "val_f1_score"
    mode: "max"
    min_delta: 0.001
  
  # Checkpointing
  checkpoint:
    save_best_only: true
    save_last: true
    monitor: "val_f1_score"
    mode: "max"
    
  # Mixed precision (for GPU training)
  mixed_precision: true
  
  # Gradient clipping
  gradient_clip_val: 1.0
```

## Cross-Validation Configuration

```yaml
cross_validation:
  n_folds: 5
  strategy: "protein_stratified"  # Options: kfold, stratified, protein_stratified
  shuffle: true
  random_state: 42
  
  # Parallel processing
  n_jobs: -1  # Number of parallel jobs
  
  # Aggregation
  aggregation_method: "mean"  # How to aggregate fold results
  report_std: true  # Whether to report standard deviations
```

## Evaluation Configuration

```yaml
evaluation:
  # Metrics to compute
  metrics:
    - "accuracy"
    - "precision" 
    - "recall"
    - "f1_score"
    - "roc_auc"
    - "pr_auc"
    - "matthews_correlation"
  
  # Threshold optimization
  optimize_threshold: true
  threshold_metric: "f1_score"  # Metric to optimize threshold for
  
  # Bootstrap confidence intervals
  bootstrap:
    enabled: true
    n_samples: 1000
    confidence_level: 0.95
  
  # Statistical testing
  statistical_tests:
    enabled: true
    tests: ["mcnemar", "wilcoxon"]
    alpha: 0.05
  
  # Visualization
  plots:
    confusion_matrix: true
    roc_curve: true
    precision_recall_curve: true
    feature_importance: true  # For tree-based models
    attention_heatmaps: true  # For transformer models
```

## Experiment Tracking (WandB)

```yaml
wandb:
  # Project configuration
  project: "phosphorylation-prediction"
  entity: "research-team"
  
  # Run configuration
  run_name: null  # Auto-generated if null
  tags: ["experiment", "baseline"]
  notes: "Baseline experiment with XGBoost"
  
  # Logging configuration
  log_frequency: 10  # Log every N steps
  save_code: true
  save_model: true
  
  # Hyperparameter sweep (optional)
  sweep:
    enabled: false
    method: "bayes"  # Options: grid, random, bayes
    metric:
      name: "val_f1_score"
      goal: "maximize"
    parameters:
      learning_rate:
        distribution: "log_uniform"
        min: 1e-5
        max: 1e-2
      batch_size:
        values: [16, 32, 64]
```

## Hardware Configuration

```yaml
hardware:
  # Device configuration
  device: "auto"  # Options: auto, cpu, cuda
  gpu_ids: [0]  # GPU IDs to use (for multi-GPU)
  
  # Memory management
  memory:
    batch_size_auto_tune: true
    max_memory_fraction: 0.8
    gradient_checkpointing: false  # Trade compute for memory
  
  # Parallelization
  num_workers: 4  # DataLoader workers
  pin_memory: true
  
  # Reproducibility
  deterministic: true
  benchmark: false  # Set to true for consistent input sizes
```

## Advanced Configuration

### Feature Engineering

```yaml
feature_engineering:
  # Sequence-based features
  sequence_features:
    length: true
    gc_content: false  # Not applicable for proteins
    
  # Position-based features
  position_features:
    relative_position: true
    distance_to_terminals: true
    
  # Evolutionary features (if available)
  evolutionary_features:
    conservation_score: false
    phylogenetic_diversity: false
  
  # Structural features (if available)  
  structural_features:
    secondary_structure: false
    solvent_accessibility: false
    disorder_score: false
```

### Custom Preprocessing

```yaml
preprocessing:
  # Sequence filtering
  min_sequence_length: 10
  max_sequence_length: 1000
  allowed_amino_acids: "ACDEFGHIKLMNPQRSTVWY"
  
  # Quality control
  remove_duplicates: true
  remove_ambiguous: true  # Remove sequences with 'X', 'B', etc.
  
  # Normalization
  normalize_features: true
  normalization_method: "standard"  # Options: standard, minmax, robust
```

## Configuration Templates

### Quick Start Template

```yaml
# Minimal configuration for quick experiments
model:
  type: "xgboost"
  n_estimators: 50

data:
  window_size: 21
  train_ratio: 0.8
  val_ratio: 0.1
  test_ratio: 0.1

training:
  epochs: 50
  batch_size: 32

evaluation:
  metrics: ["accuracy", "f1_score", "roc_auc"]
```

### Research Template

```yaml
# Comprehensive configuration for research experiments
model:
  type: "ensemble"
  ensemble_type: "stacking"
  base_models:
    - type: "xgboost"
      config:
        n_estimators: 200
        max_depth: 8
    - type: "transformer"
      config:
        model_name: "facebook/esm2_t12_35M_UR50D"

data:
  window_size: 21
  split_strategy: "protein_level"
  balance_classes: true

training:
  epochs: 200
  early_stopping:
    enabled: true
    patience: 20

cross_validation:
  n_folds: 10
  strategy: "protein_stratified"

evaluation:
  metrics: ["accuracy", "precision", "recall", "f1_score", "roc_auc", "pr_auc"]
  bootstrap:
    enabled: true
    n_samples: 1000

wandb:
  project: "phosphorylation-research"
  save_model: true
```

## Configuration Validation

The framework automatically validates configurations and will report errors for:

- Invalid parameter values
- Missing required fields
- Incompatible parameter combinations
- Hardware compatibility issues

To validate a configuration without running an experiment:

```python
from config import load_config, validate_config

config = load_config("my_config.yaml")
is_valid, errors = validate_config(config)

if not is_valid:
    for error in errors:
        print(f"Error: {error}")
```

## Best Practices

1. **Start Simple**: Begin with basic configurations and gradually add complexity
2. **Use Templates**: Start from provided templates for common use cases
3. **Version Control**: Keep configuration files in version control
4. **Document Changes**: Add comments to explain custom parameters
5. **Validate Early**: Test configurations on small datasets first
6. **Monitor Resources**: Adjust batch sizes and workers based on available hardware

## Common Configuration Patterns

### High Performance Setup
- Use ensemble models
- Enable mixed precision training
- Optimize batch sizes for hardware
- Use multiple GPUs if available

### Memory Constrained Setup
- Reduce batch sizes
- Use gradient checkpointing
- Disable TPC features
- Use smaller transformer models

### Quick Prototyping Setup
- Use XGBoost models
- Reduce number of estimators
- Use smaller validation sets
- Disable expensive evaluations