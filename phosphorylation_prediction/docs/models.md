# Model Guide

This guide provides detailed information about the different model types available in the Phosphorylation Site Prediction Framework.

## Overview

The framework supports three main categories of models:

1. **XGBoost Models**: Feature-based gradient boosting
2. **Transformer Models**: Deep learning with protein language models
3. **Ensemble Models**: Combinations of multiple models

Each model type has different strengths and is suitable for different scenarios.

## XGBoost Models

### Overview

XGBoost (eXtreme Gradient Boosting) models use extracted features from protein sequences to make predictions. They are fast, interpretable, and work well with limited data.

### Features Used

The XGBoost model extracts multiple types of features:

#### Amino Acid Composition (AAC)
- Frequency of each of the 20 amino acids
- 20-dimensional feature vector
- Captures overall amino acid preferences

```python
# Example: Sequence "MKWVT" 
# AAC features: [0.0, 0.0, 0.0, ..., 0.2, 0.2, 0.2, 0.2, 0.2, ...]
#                A    C    D         K    M    T    V    W
```

#### Dipeptide Composition (DPC)
- Frequency of all possible amino acid pairs
- 400-dimensional feature vector (20×20)
- Captures local sequence patterns

```python
# Example: Sequence "MKWVT"
# Dipeptides: MK, KW, WV, VT
# DPC features: [0.0, ..., 0.25, ..., 0.25, ..., 0.25, ..., 0.25, ...]
```

#### Tripeptide Composition (TPC)
- Frequency of all possible amino acid triplets
- 8000-dimensional feature vector (20×20×20)
- Captures more complex sequence patterns
- Memory intensive - use with caution

#### Binary Encoding
- One-hot encoding of amino acids in the sequence window
- Window_size × 20 dimensional features
- Preserves positional information

#### Physicochemical Properties
- Hydrophobicity, charge, molecular weight, etc.
- Aggregated statistics (mean, std, min, max) across the window
- Captures biochemical properties

### Configuration

```yaml
model:
  type: "xgboost"
  
  # XGBoost hyperparameters
  n_estimators: 100
  max_depth: 6
  learning_rate: 0.1
  subsample: 0.8
  colsample_bytree: 0.8
  
  # Feature selection
  features:
    use_aac: true
    use_dpc: true  
    use_tpc: false  # Memory intensive
    use_binary: true
    use_physicochemical: true
  
  # Regularization
  reg_alpha: 0.0  # L1 regularization
  reg_lambda: 1.0  # L2 regularization
  
  # Performance
  random_state: 42
  n_jobs: -1
```

### Advantages
- Fast training and prediction
- Interpretable feature importance
- Works well with small datasets
- No GPU required
- Robust to overfitting

### Disadvantages
- Requires manual feature engineering
- May miss complex sequence patterns
- Limited by feature representation

### Best Use Cases
- Quick prototyping
- Limited computational resources
- Interpretability requirements
- Small to medium datasets

## Transformer Models

### Overview

Transformer models use pre-trained protein language models (ESM2) to understand protein sequences. They can capture complex patterns and relationships without manual feature engineering.

### Architecture

The framework uses ESM2 (Evolutionary Scale Modeling) models with custom classification heads:

1. **ESM2 Backbone**: Pre-trained protein language model
2. **Context Aggregation**: Combines information from sequence context
3. **Classification Head**: Maps to phosphorylation predictions

### ESM2 Model Variants

| Model | Parameters | Embedding Dim | Description |
|-------|------------|---------------|-------------|
| esm2_t6_8M_UR50D | 8M | 320 | Smallest, fastest |
| esm2_t12_35M_UR50D | 35M | 480 | Good balance |
| esm2_t30_150M_UR50D | 150M | 640 | Larger, more accurate |
| esm2_t33_650M_UR50D | 650M | 1280 | Largest, best performance |

### Context Aggregation Methods

#### Mean Pooling
```python
# Average embeddings across sequence positions
context_vector = torch.mean(sequence_embeddings, dim=1)
```

#### Max Pooling
```python
# Take maximum values across sequence positions
context_vector = torch.max(sequence_embeddings, dim=1)[0]
```

#### Attention Pooling
```python
# Learned attention weights for each position
attention_weights = attention_layer(sequence_embeddings)
context_vector = torch.sum(attention_weights * sequence_embeddings, dim=1)
```

### Configuration

```yaml
model:
  type: "transformer"
  
  # Model selection
  model_name: "facebook/esm2_t12_35M_UR50D"
  
  # Architecture
  num_labels: 2
  hidden_dropout_prob: 0.1
  attention_probs_dropout_prob: 0.1
  
  # Context aggregation
  context_aggregation: "attention"  # mean, max, attention
  freeze_backbone: false  # Whether to freeze ESM2 weights
  
  # Training
  learning_rate: 2e-5
  weight_decay: 0.01
  warmup_steps: 500
  
  # Optimization
  use_mixed_precision: true
  gradient_accumulation_steps: 1
  gradient_checkpointing: false
```

### Training Process

1. **Tokenization**: Convert amino acid sequences to tokens
2. **Embedding**: ESM2 generates contextual embeddings
3. **Aggregation**: Combine embeddings using chosen method
4. **Classification**: Map to phosphorylation probability
5. **Fine-tuning**: Update weights based on task-specific data

### Advantages
- Captures complex sequence patterns
- No manual feature engineering
- State-of-the-art performance
- Transfer learning from large datasets
- Attention visualization

### Disadvantages
- Requires GPU for practical training
- Longer training time
- More memory intensive
- Less interpretable
- Risk of overfitting on small datasets

### Best Use Cases
- Large datasets
- GPU resources available
- Maximum performance required
- Complex sequence patterns expected

## Ensemble Models

### Overview

Ensemble models combine predictions from multiple base models to achieve better performance than individual models. The framework supports three ensemble strategies.

### Voting Ensemble

Combines predictions through weighted voting.

#### Soft Voting
```python
# Combine prediction probabilities
final_prob = w1 * model1_prob + w2 * model2_prob + w3 * model3_prob
```

#### Hard Voting
```python
# Combine binary predictions
final_pred = majority_vote([model1_pred, model2_pred, model3_pred])
```

#### Configuration
```yaml
model:
  type: "ensemble"
  ensemble_type: "voting"
  
  base_models:
    - type: "xgboost"
      weight: 1.0
      config:
        n_estimators: 100
    - type: "transformer" 
      weight: 1.5  # Higher weight for better model
      config:
        model_name: "facebook/esm2_t12_35M_UR50D"
  
  voting_strategy: "soft"  # soft or hard
  optimize_weights: true  # Learn optimal weights
```

### Stacking Ensemble

Uses a meta-learner to combine base model predictions.

#### Architecture
1. **Level 0**: Base models make predictions
2. **Level 1**: Meta-learner combines base predictions
3. **Cross-validation**: Prevents overfitting

#### Configuration
```yaml
model:
  type: "ensemble"
  ensemble_type: "stacking"
  
  base_models:
    - type: "xgboost"
    - type: "transformer"
  
  meta_learner:
    type: "logistic_regression"  # or random_forest, neural_network
    C: 1.0
    
  cv_folds: 5  # For generating meta-features
```

### Dynamic Ensemble

Selects the best model for each prediction based on local competence.

#### Competence Methods

**Local Accuracy**
```python
# Use k nearest neighbors to estimate local performance
competence = accuracy_in_neighborhood(sample, k=5)
```

**Overall Local Accuracy (OLA)**
```python
# Weight by distance to neighbors
competence = weighted_accuracy_in_neighborhood(sample, k=5)
```

**Dynamic Classifier Selection (DCS)**
```python
# Select single best classifier for each sample
best_classifier = argmax(competence_scores)
```

#### Configuration
```yaml
model:
  type: "ensemble"
  ensemble_type: "dynamic"
  
  base_models:
    - type: "xgboost"
    - type: "transformer"
  
  competence_method: "local_accuracy"  # ola, dcs
  k_neighbors: 5
  selection_method: "best"  # best, all_competent
```

## Model Selection Guidelines

### Dataset Size
- **Small (< 1K samples)**: XGBoost
- **Medium (1K - 10K samples)**: XGBoost or small Transformer
- **Large (> 10K samples)**: Transformer or Ensemble

### Computational Resources
- **CPU only**: XGBoost
- **Limited GPU**: Small Transformer (esm2_t6_8M)
- **High-end GPU**: Large Transformer or Ensemble

### Performance Requirements
- **Good enough**: XGBoost
- **Best possible**: Ensemble with large Transformer
- **Balanced**: Medium Transformer

### Interpretability Requirements
- **High**: XGBoost with feature importance
- **Medium**: XGBoost + Attention visualization
- **Low**: Any model type

### Training Time Constraints
- **Minutes**: XGBoost
- **Hours**: Transformer
- **Days**: Complex Ensemble

## Model Implementation Details

### XGBoost Implementation

```python
from models import XGBoostModel

# Initialize model
model = XGBoostModel(config)

# Train model
model.fit(X_train, y_train, X_val, y_val)

# Get feature importance
importance = model.get_feature_importance()

# Make predictions
predictions = model.predict_proba(X_test)
```

### Transformer Implementation

```python
from models import TransformerModel

# Initialize model
model = TransformerModel(config)

# Train model with gradient accumulation
model.fit(train_dataloader, val_dataloader)

# Extract attention weights
attention_weights = model.get_attention_weights(sequences)

# Make predictions
predictions = model.predict_proba(test_dataloader)
```

### Ensemble Implementation

```python
from models import VotingEnsemble

# Initialize ensemble
ensemble = VotingEnsemble(config)

# Train base models and ensemble
ensemble.fit(X_train, y_train, X_val, y_val)

# Get individual model predictions
individual_preds = ensemble.get_individual_predictions(X_test)

# Make ensemble predictions
ensemble_preds = ensemble.predict_proba(X_test)
```

## Performance Optimization

### XGBoost Optimization
- Use early stopping
- Tune regularization parameters
- Optimize feature selection
- Use parallel processing

### Transformer Optimization
- Use mixed precision training
- Gradient checkpointing for memory
- Optimal batch size for GPU
- Learning rate scheduling

### Ensemble Optimization
- Diverse base models
- Optimal weight learning
- Cross-validation for meta-learning
- Efficient prediction caching

## Model Evaluation

### Metrics by Model Type

**XGBoost**
- Feature importance analysis
- Partial dependence plots
- SHAP values for interpretability

**Transformer**
- Attention visualization
- Layer-wise analysis
- Embedding similarity analysis

**Ensemble**
- Individual vs. ensemble performance
- Model diversity metrics
- Contribution analysis

### Validation Strategies

**All Models**
- Cross-validation with protein-level splitting
- Bootstrap confidence intervals
- Statistical significance testing

**Model-Specific**
- Feature ablation (XGBoost)
- Attention analysis (Transformer)
- Base model correlation (Ensemble)

## Troubleshooting

### Common Issues

**XGBoost**
- Overfitting: Increase regularization, reduce n_estimators
- Poor performance: Check feature engineering, try different parameters
- Memory issues: Disable TPC features, reduce data size

**Transformer**
- OOM errors: Reduce batch size, enable gradient checkpointing
- Slow convergence: Adjust learning rate, increase warmup steps
- Poor transfer: Try different ESM2 variants, check data quality

**Ensemble**
- Base model correlation: Ensure diverse models
- Meta-learner overfitting: Use more CV folds, simpler meta-learner
- Computational cost: Cache base predictions, parallel training