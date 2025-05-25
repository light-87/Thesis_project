# Phosphorylation Site Prediction Framework

A standardized, modular framework for predicting phosphorylation sites in protein sequences using machine learning approaches including XGBoost, Transformer models, and ensemble methods.

## Features

- **Multiple Model Types**: XGBoost, Transformer (ESM2-based), and Ensemble models
- **Comprehensive Feature Extraction**: Amino acid composition, dipeptide composition, tripeptide composition, binary encoding, and physicochemical properties
- **Advanced Ensemble Methods**: Voting, stacking, and dynamic selection
- **Experiment Tracking**: Integration with Weights & Biases (WandB)
- **Cross-Validation**: Protein-level splitting to avoid data leakage
- **Memory Optimization**: GPU memory management and batch size estimation
- **Reproducibility**: Comprehensive seed management and environment tracking
- **Flexible Configuration**: YAML-based configuration system

## Installation

### Prerequisites

- Python 3.8 or higher
- CUDA-compatible GPU (optional, for transformer models)

### Install from Source

```bash
git clone https://github.com/research-team/phosphorylation-prediction.git
cd phosphorylation-prediction
pip install -e .
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

For GPU support:
```bash
pip install -e .[gpu]
```

For development:
```bash
pip install -e .[dev]
```

## Quick Start

### 1. Prepare Your Data

Ensure your data is in CSV format with columns:
- `sequence`: Protein sequence
- `label`: Binary label (1 for phosphorylation site, 0 for non-site)
- `position`: Position in the sequence (optional)

Example:
```csv
sequence,label,position
MKWVTFISLLLLFSSAYSRGV,1,5
ACDEFGHIKLMNPQRSTVWYK,0,10
```

### 2. Configure Your Experiment

Create a configuration file or use the default:

```bash
cp config/default_config.yaml my_experiment_config.yaml
```

Edit the configuration as needed:
```yaml
# Model configuration
model:
  type: "xgboost"  # or "transformer" or "ensemble"
  
# Data configuration
data:
  window_size: 21
  train_ratio: 0.7
  val_ratio: 0.15
  test_ratio: 0.15

# Training configuration
training:
  epochs: 100
  batch_size: 32
  learning_rate: 0.001
```

### 3. Train a Model

```bash
python scripts/train.py \
    --config my_experiment_config.yaml \
    --experiment-type single \
    --model-type xgboost \
    --data-path data/sequences.csv \
    --output-dir experiments/xgboost_run1
```

### 4. Make Predictions

```bash
python scripts/predict.py \
    --model-path experiments/xgboost_run1/checkpoints/experiment_checkpoint.json \
    --input-file new_sequences.csv \
    --output-file predictions.csv
```

### 5. Analyze Results

```bash
python scripts/analyze.py \
    --experiment-dir experiments/xgboost_run1 \
    --analysis-type all
```

## Usage Examples

### Training Different Model Types

#### XGBoost Model
```bash
python scripts/train.py \
    --config config/xgboost_config.yaml \
    --experiment-type single \
    --model-type xgboost \
    --data-path data/train.csv \
    --output-dir experiments/xgboost
```

#### Transformer Model
```bash
python scripts/train.py \
    --config config/transformer_config.yaml \
    --experiment-type single \
    --model-type transformer \
    --data-path data/train.csv \
    --output-dir experiments/transformer
```

#### Ensemble Model
```bash
python scripts/train.py \
    --config config/ensemble_config.yaml \
    --experiment-type ensemble \
    --data-path data/train.csv \
    --output-dir experiments/ensemble
```

### Cross-Validation
```bash
python scripts/train.py \
    --config config/cv_config.yaml \
    --experiment-type cross_validation \
    --model-type xgboost \
    --data-path data/train.csv \
    --output-dir experiments/cv_xgboost
```

### Comparing Multiple Experiments
```bash
python scripts/analyze.py \
    --experiment-dir experiments/xgboost \
    --compare-experiments experiments/transformer experiments/ensemble \
    --analysis-type comparison
```

## Project Structure

```
phosphorylation_prediction/
├── config/                 # Configuration files
│   ├── default_config.yaml
│   ├── xgboost_config.yaml
│   ├── transformer_config.yaml
│   └── ensemble_config.yaml
├── src/                    # Source code
│   ├── config/            # Configuration management
│   ├── data/              # Data processing
│   ├── models/            # Model implementations
│   ├── training/          # Training utilities
│   ├── evaluation/        # Evaluation and metrics
│   ├── experiments/       # Experiment framework
│   └── utils/             # Utility modules
├── scripts/               # Command-line scripts
│   ├── train.py          # Training script
│   ├── predict.py        # Prediction script
│   └── analyze.py        # Analysis script
├── tests/                 # Test files
├── docs/                  # Documentation
└── experiments/           # Experiment outputs
```

## Configuration

The framework uses YAML configuration files for experiment setup. Key configuration sections:

- **model**: Model type and hyperparameters
- **data**: Data processing parameters
- **training**: Training configuration
- **evaluation**: Evaluation metrics and settings
- **wandb**: Experiment tracking configuration

See `config/default_config.yaml` for a complete example.

## Models

### XGBoost Model
- Gradient boosting classifier
- Feature-based approach using extracted protein sequence features
- Fast training and prediction
- Interpretable feature importance

### Transformer Model
- Based on ESM2 protein language model
- Attention mechanism for sequence context
- State-of-the-art performance on protein tasks
- Requires GPU for efficient training

### Ensemble Models
- **Voting Ensemble**: Weighted majority voting
- **Stacking Ensemble**: Meta-learner on base model predictions
- **Dynamic Ensemble**: Competence-based model selection

## Features

The framework extracts multiple types of features from protein sequences:

- **Amino Acid Composition (AAC)**: Frequency of each amino acid
- **Dipeptide Composition (DPC)**: Frequency of amino acid pairs
- **Tripeptide Composition (TPC)**: Frequency of amino acid triplets
- **Binary Encoding**: One-hot encoding of amino acids
- **Physicochemical Properties**: Hydrophobicity, charge, etc.

## Evaluation

Comprehensive evaluation metrics:
- Accuracy, Precision, Recall, F1-score
- ROC AUC and PR AUC
- Confusion matrix
- Bootstrap confidence intervals
- Statistical significance testing

## Experiment Tracking

Integration with Weights & Biases for:
- Hyperparameter logging
- Metric tracking
- Model versioning
- Visualization
- Collaborative experiments

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests for new functionality
5. Run the test suite
6. Submit a pull request

## Development

### Running Tests
```bash
pytest tests/
```

### Code Formatting
```bash
black src/ scripts/ tests/
isort src/ scripts/ tests/
```

### Type Checking
```bash
mypy src/
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Citation

If you use this framework in your research, please cite:

```bibtex
@software{phosphorylation_prediction_framework,
  title={Phosphorylation Site Prediction Framework},
  author={Research Team},
  year={2024},
  url={https://github.com/research-team/phosphorylation-prediction}
}
```

## Support

For questions and support:
- Open an issue on GitHub
- Email: research@university.edu
- Documentation: https://phosphorylation-prediction.readthedocs.io/

## Changelog

### Version 1.0.0
- Initial release
- XGBoost, Transformer, and Ensemble models
- Comprehensive evaluation framework
- WandB integration
- Cross-validation support