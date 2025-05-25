# Getting Started

This guide will help you get up and running with the Phosphorylation Site Prediction Framework.

## Installation

### Prerequisites

Before installing the framework, ensure you have:

- Python 3.8 or higher
- pip package manager
- Git (for cloning the repository)
- CUDA-compatible GPU (optional, for transformer models)

### Step 1: Clone the Repository

```bash
git clone https://github.com/research-team/phosphorylation-prediction.git
cd phosphorylation-prediction
```

### Step 2: Install Dependencies

For basic usage:
```bash
pip install -r requirements.txt
```

For development:
```bash
pip install -e .[dev]
```

For GPU support:
```bash
pip install -e .[gpu]
```

### Step 3: Verify Installation

Test that everything is working:
```bash
python -c "import src.config; print('Installation successful!')"
```

## Data Preparation

### Input Format

The framework expects data in CSV format with the following columns:

- **sequence**: Protein sequence (required)
- **label**: Binary label (1 for phosphorylation site, 0 for non-site) (required)
- **position**: Position in the sequence (optional)
- **protein_id**: Protein identifier (optional)

Example CSV:
```csv
sequence,label,position,protein_id
MKWVTFISLLLLFSSAYSRGV,1,5,P12345
ACDEFGHIKLMNPQRSTVWYK,0,10,P67890
MKQHKAMIVALIVICITAVVAAL,1,8,P13579
```

### Data Quality Requirements

- Sequences should contain only standard amino acid letters (A-Z)
- Window extraction will be performed automatically based on the configured window size
- Sequences shorter than the window size will be padded
- Labels should be 0 or 1

### Example Data Preparation Script

```python
import pandas as pd

# Load your raw data
data = pd.read_csv("raw_data.csv")

# Ensure proper format
data = data[['sequence', 'label']].copy()
data = data.dropna()  # Remove missing values
data['label'] = data['label'].astype(int)  # Ensure integer labels

# Save prepared data
data.to_csv("prepared_data.csv", index=False)
```

## Basic Usage

### 1. Configure Your Experiment

Start with the default configuration:
```bash
cp config/default_config.yaml my_config.yaml
```

Edit key parameters:
```yaml
# Model configuration
model:
  type: "xgboost"  # Choose: xgboost, transformer, ensemble
  
# Data configuration
data:
  window_size: 21  # Sequence window size
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# Training configuration
training:
  epochs: 100
  batch_size: 32
```

### 2. Train Your First Model

```bash
python scripts/train.py \
    --config my_config.yaml \
    --experiment-type single \
    --model-type xgboost \
    --data-path prepared_data.csv \
    --output-dir experiments/my_first_experiment
```

### 3. Make Predictions

```bash
python scripts/predict.py \
    --model-path experiments/my_first_experiment/checkpoints/experiment_checkpoint.json \
    --input-file new_sequences.csv \
    --output-file predictions.csv
```

### 4. Analyze Results

```bash
python scripts/analyze.py \
    --experiment-dir experiments/my_first_experiment \
    --analysis-type all
```

## Understanding the Output

### Training Output

After training, you'll find:
- `checkpoints/`: Model checkpoints and experiment state
- `logs/`: Training logs and metrics
- `plots/`: Training curves and visualizations
- `config.yaml`: Final configuration used
- `metrics.json`: Final evaluation metrics

### Prediction Output

The prediction file contains:
- `sequence_id`: Input sequence identifier
- `sequence`: Original sequence
- `position`: Position in sequence
- `window`: Extracted sequence window
- `probability`: Prediction probability (0-1)
- `prediction`: Binary prediction (0 or 1)
- `confident`: Whether prediction is confident (Yes/No)

### Analysis Output

Analysis generates:
- `metrics_summary.csv`: Comprehensive metrics
- `confusion_matrix.png`: Confusion matrix visualization
- `roc_curve.png`: ROC curve plot
- `feature_importance.png`: Feature importance (for XGBoost)
- `analysis_report.txt`: Detailed text report

## Common Workflows

### Workflow 1: Single Model Training

1. Prepare data in CSV format
2. Create/modify configuration file
3. Train model with `train.py`
4. Evaluate with `analyze.py`
5. Make predictions with `predict.py`

### Workflow 2: Model Comparison

1. Train multiple models with different configurations
2. Use `analyze.py` with `--compare-experiments` to compare
3. Select best performing model

### Workflow 3: Cross-Validation

1. Set up cross-validation configuration
2. Run with `--experiment-type cross_validation`
3. Analyze aggregated results

## Tips for Success

### Data Tips
- Ensure balanced datasets when possible
- Use protein-level splitting for cross-validation
- Validate data quality before training

### Training Tips
- Start with XGBoost for quick experiments
- Use GPU for transformer models
- Monitor training with WandB integration

### Performance Tips
- Adjust batch size based on available memory
- Use early stopping to prevent overfitting
- Experiment with different window sizes

## Troubleshooting

### Common Issues

**ImportError: Module not found**
- Ensure all dependencies are installed
- Check Python path includes src/ directory

**CUDA out of memory**
- Reduce batch size in configuration
- Use memory optimization utilities
- Consider using CPU for small datasets

**Poor model performance**
- Check data quality and balance
- Experiment with different features
- Try ensemble methods

**Configuration errors**
- Validate YAML syntax
- Check required fields are present
- Use default config as template

### Getting Help

1. Check the documentation in `docs/`
2. Look at example configurations in `config/`
3. Examine the source code in `src/`
4. Open an issue on GitHub
5. Contact the development team

## Next Steps

Now that you have the basics working:

1. Read the [Configuration Guide](configuration.md) for detailed parameter tuning
2. Check the [Model Guide](models.md) to understand different model types
3. Explore [Advanced Usage](advanced_usage.md) for complex workflows
4. Review [API Reference](api_reference.md) for programmatic usage