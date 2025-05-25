# Advanced Usage Guide

This guide covers advanced features and workflows for the Phosphorylation Site Prediction Framework.

## Programmatic API Usage

### Basic API Usage

Instead of using command-line scripts, you can use the framework programmatically:

```python
import sys
sys.path.append("src")

from config import load_config
from data import SequenceDataset, SequenceProcessor, FeatureExtractor
from models import XGBoostModel, TransformerModel, VotingEnsemble
from training import Trainer
from evaluation import ModelEvaluator
from utils import set_seed, get_logger

# Setup
set_seed(42)
logger = get_logger(__name__)

# Load configuration
config = load_config("config/default_config.yaml")

# Load and process data
dataset = SequenceDataset.from_csv("data/sequences.csv")
processor = SequenceProcessor(config['data'])
train_data, val_data, test_data = dataset.split(config['data'])

# Initialize model
model = XGBoostModel(config['model'])

# Train model
trainer = Trainer(config['training'])
trainer.fit(model, train_data, val_data)

# Evaluate model
evaluator = ModelEvaluator(config['evaluation'])
results = evaluator.evaluate(model, test_data)

print(f"Test accuracy: {results['accuracy']:.4f}")
```

### Custom Data Loading

```python
from data import SequenceDataset
import pandas as pd

# Load from multiple sources
df1 = pd.read_csv("dataset1.csv")
df2 = pd.read_csv("dataset2.csv")
combined_df = pd.concat([df1, df2], ignore_index=True)

# Create dataset with custom processing
dataset = SequenceDataset(
    sequences=combined_df['sequence'].tolist(),
    labels=combined_df['label'].tolist(),
    metadata={
        'protein_ids': combined_df['protein_id'].tolist(),
        'positions': combined_df['position'].tolist()
    }
)

# Custom train/test split
train_proteins = set(combined_df['protein_id'].iloc[:1000])
test_proteins = set(combined_df['protein_id'].iloc[1000:])

train_mask = combined_df['protein_id'].isin(train_proteins)
test_mask = combined_df['protein_id'].isin(test_proteins)

train_dataset = SequenceDataset(
    sequences=combined_df[train_mask]['sequence'].tolist(),
    labels=combined_df[train_mask]['label'].tolist()
)

test_dataset = SequenceDataset(
    sequences=combined_df[test_mask]['sequence'].tolist(),
    labels=combined_df[test_mask]['label'].tolist()
)
```

### Custom Model Implementation

```python
from models.base_model import BaseModel
import numpy as np
from sklearn.ensemble import RandomForestClassifier

class CustomRandomForestModel(BaseModel):
    """Custom Random Forest model implementation."""
    
    def __init__(self, config):
        super().__init__(config)
        self.model = RandomForestClassifier(**config)
        self.feature_extractor = FeatureExtractor()
    
    def fit(self, X_train, y_train, X_val=None, y_val=None, **kwargs):
        # Extract features
        if isinstance(X_train[0], str):
            X_train = self.feature_extractor.extract_features(X_train)
        
        # Train model
        self.model.fit(X_train, y_train)
        
        # Validation evaluation
        if X_val is not None and y_val is not None:
            if isinstance(X_val[0], str):
                X_val = self.feature_extractor.extract_features(X_val)
            val_score = self.model.score(X_val, y_val)
            print(f"Validation accuracy: {val_score:.4f}")
    
    def predict(self, X):
        if isinstance(X[0], str):
            X = self.feature_extractor.extract_features(X)
        return self.model.predict(X)
    
    def predict_proba(self, X):
        if isinstance(X[0], str):
            X = self.feature_extractor.extract_features(X)
        return self.model.predict_proba(X)
    
    def get_feature_importance(self):
        return self.model.feature_importances_

# Usage
config = {'n_estimators': 100, 'max_depth': 10, 'random_state': 42}
model = CustomRandomForestModel(config)
```

## Custom Experiments

### Experiment Base Class Extension

```python
from experiments.base_experiment import BaseExperiment
import pandas as pd

class CustomExperiment(BaseExperiment):
    """Custom experiment with specific data processing."""
    
    def load_data(self, data_path):
        """Custom data loading logic."""
        df = pd.read_csv(data_path)
        
        # Custom preprocessing
        df = df[df['sequence_length'] >= 20]  # Filter short sequences
        df = df[df['organism'] == 'human']     # Filter by organism
        
        # Balance classes
        positive_samples = df[df['label'] == 1]
        negative_samples = df[df['label'] == 0].sample(len(positive_samples))
        df = pd.concat([positive_samples, negative_samples])
        
        self.data = df
        return df
    
    def preprocess_data(self, data):
        """Custom preprocessing logic."""
        # Custom window extraction
        processor = SequenceProcessor(
            window_size=self.config['data']['window_size'],
            custom_padding='N'  # Use 'N' for padding
        )
        
        windows = []
        labels = []
        
        for _, row in data.iterrows():
            seq_windows, positions = processor.extract_windows(row['sequence'])
            window_labels = [row['label']] * len(seq_windows)
            
            windows.extend(seq_windows)
            labels.extend(window_labels)
        
        return windows, labels
    
    def train_models(self, train_data, val_data):
        """Custom training logic."""
        # Train multiple models with different configurations
        models = {}
        
        for model_type in ['xgboost', 'transformer']:
            config = self.config['model'].copy()
            config['type'] = model_type
            
            if model_type == 'xgboost':
                model = XGBoostModel(config)
            else:
                model = TransformerModel(config)
            
            # Custom training with callbacks
            trainer = Trainer(self.config['training'])
            trainer.add_callback('early_stopping', patience=20)
            trainer.add_callback('lr_scheduler', factor=0.5)
            
            trainer.fit(model, train_data, val_data)
            models[model_type] = model
        
        self.models = models
        return models
    
    def evaluate_models(self, test_data):
        """Custom evaluation logic."""
        results = {}
        
        for name, model in self.models.items():
            evaluator = ModelEvaluator(self.config['evaluation'])
            
            # Custom metrics
            evaluator.add_metric('custom_specificity', self._calculate_specificity)
            evaluator.add_metric('custom_sensitivity', self._calculate_sensitivity)
            
            model_results = evaluator.evaluate(model, test_data)
            results[name] = model_results
        
        # Cross-model analysis
        results['ensemble'] = self._evaluate_ensemble(test_data)
        
        return results
    
    def _calculate_specificity(self, y_true, y_pred):
        tn = ((y_true == 0) & (y_pred == 0)).sum()
        fp = ((y_true == 0) & (y_pred == 1)).sum()
        return tn / (tn + fp) if (tn + fp) > 0 else 0
    
    def _calculate_sensitivity(self, y_true, y_pred):
        tp = ((y_true == 1) & (y_pred == 1)).sum()
        fn = ((y_true == 1) & (y_pred == 0)).sum()
        return tp / (tp + fn) if (tp + fn) > 0 else 0
    
    def _evaluate_ensemble(self, test_data):
        # Create and evaluate ensemble
        ensemble = VotingEnsemble({
            'base_models': [
                {'type': 'xgboost', 'weight': 1.0},
                {'type': 'transformer', 'weight': 1.5}
            ]
        })
        
        # Use pre-trained models
        ensemble.models = self.models
        
        evaluator = ModelEvaluator(self.config['evaluation'])
        return evaluator.evaluate(ensemble, test_data)

# Usage
experiment = CustomExperiment(config, output_dir="custom_experiment")
experiment.run()
```

## Hyperparameter Optimization

### Manual Grid Search

```python
from itertools import product
import json

def grid_search_xgboost(train_data, val_data, param_grid):
    """Manual grid search for XGBoost hyperparameters."""
    
    best_score = 0
    best_params = None
    results = []
    
    # Generate all parameter combinations
    param_names = list(param_grid.keys())
    param_values = list(param_grid.values())
    
    for combination in product(*param_values):
        params = dict(zip(param_names, combination))
        
        # Train model with current parameters
        config = {'type': 'xgboost', **params}
        model = XGBoostModel(config)
        model.fit(train_data[0], train_data[1], val_data[0], val_data[1])
        
        # Evaluate
        y_pred = model.predict_proba(val_data[0])[:, 1]
        score = f1_score(val_data[1], (y_pred > 0.5).astype(int))
        
        results.append({
            'params': params,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = params
        
        print(f"Params: {params}, Score: {score:.4f}")
    
    return best_params, best_score, results

# Usage
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 6, 9],
    'learning_rate': [0.01, 0.1, 0.2]
}

best_params, best_score, all_results = grid_search_xgboost(
    train_data, val_data, param_grid
)

print(f"Best parameters: {best_params}")
print(f"Best score: {best_score:.4f}")
```

### WandB Hyperparameter Sweeps

```python
import wandb

def train_with_wandb():
    """Training function for WandB sweeps."""
    
    # Initialize wandb run
    wandb.init()
    
    # Get hyperparameters from wandb
    config = wandb.config
    
    # Convert wandb config to framework config
    model_config = {
        'type': 'xgboost',
        'n_estimators': config.n_estimators,
        'max_depth': config.max_depth,
        'learning_rate': config.learning_rate,
        'subsample': config.subsample
    }
    
    # Train model
    model = XGBoostModel(model_config)
    model.fit(train_data[0], train_data[1], val_data[0], val_data[1])
    
    # Evaluate and log
    y_pred = model.predict_proba(val_data[0])[:, 1]
    f1 = f1_score(val_data[1], (y_pred > 0.5).astype(int))
    accuracy = accuracy_score(val_data[1], (y_pred > 0.5).astype(int))
    
    wandb.log({
        'val_f1': f1,
        'val_accuracy': accuracy
    })

# Sweep configuration
sweep_config = {
    'method': 'bayes',
    'metric': {
        'name': 'val_f1',
        'goal': 'maximize'
    },
    'parameters': {
        'n_estimators': {
            'values': [50, 100, 150, 200]
        },
        'max_depth': {
            'min': 3,
            'max': 10
        },
        'learning_rate': {
            'distribution': 'log_uniform',
            'min': 0.01,
            'max': 0.3
        },
        'subsample': {
            'distribution': 'uniform',
            'min': 0.6,
            'max': 1.0
        }
    }
}

# Start sweep
sweep_id = wandb.sweep(sweep_config, project="phosphorylation-hpo")
wandb.agent(sweep_id, train_with_wandb, count=50)
```

## Custom Feature Engineering

### Advanced Feature Extraction

```python
from data.preprocessor import FeatureExtractor
import numpy as np

class AdvancedFeatureExtractor(FeatureExtractor):
    """Extended feature extractor with custom features."""
    
    def __init__(self):
        super().__init__()
        self.custom_features = {}
    
    def extract_features(self, sequences):
        """Extract comprehensive features including custom ones."""
        
        # Get base features
        features = super().extract_features(sequences)
        
        # Add custom features
        custom_features = []
        
        for seq in sequences:
            seq_features = []
            
            # Sequence length features
            seq_features.append(len(seq))
            seq_features.append(len(seq) / 21)  # Normalized by window size
            
            # Compositional features
            seq_features.extend(self._calculate_charge_features(seq))
            seq_features.extend(self._calculate_hydrophobicity_features(seq))
            seq_features.extend(self._calculate_secondary_structure_propensity(seq))
            
            # Positional features
            seq_features.extend(self._calculate_positional_features(seq))
            
            # Evolutionary features (if available)
            seq_features.extend(self._calculate_conservation_features(seq))
            
            custom_features.append(seq_features)
        
        custom_features = np.array(custom_features)
        
        # Combine with base features
        if features.size > 0:
            features = np.hstack([features, custom_features])
        else:
            features = custom_features
        
        return features
    
    def _calculate_charge_features(self, sequence):
        """Calculate charge-related features."""
        positive_aa = 'KRH'
        negative_aa = 'DE'
        
        pos_count = sum(1 for aa in sequence if aa in positive_aa)
        neg_count = sum(1 for aa in sequence if aa in negative_aa)
        
        return [
            pos_count / len(sequence),  # Positive charge density
            neg_count / len(sequence),  # Negative charge density
            (pos_count - neg_count) / len(sequence),  # Net charge
            pos_count + neg_count  # Total charged residues
        ]
    
    def _calculate_hydrophobicity_features(self, sequence):
        """Calculate hydrophobicity features."""
        # Kyte-Doolittle hydrophobicity scale
        hydrophobicity = {
            'A': 1.8, 'R': -4.5, 'N': -3.5, 'D': -3.5, 'C': 2.5,
            'Q': -3.5, 'E': -3.5, 'G': -0.4, 'H': -3.2, 'I': 4.5,
            'L': 3.8, 'K': -3.9, 'M': 1.9, 'F': 2.8, 'P': -1.6,
            'S': -0.8, 'T': -0.7, 'W': -0.9, 'Y': -1.3, 'V': 4.2
        }
        
        values = [hydrophobicity.get(aa, 0) for aa in sequence]
        
        return [
            np.mean(values),
            np.std(values),
            np.min(values),
            np.max(values),
            sum(1 for v in values if v > 0) / len(values)  # Hydrophobic fraction
        ]
    
    def _calculate_secondary_structure_propensity(self, sequence):
        """Calculate secondary structure propensities."""
        # Simplified propensities (Chou-Fasman)
        helix_prop = {
            'A': 1.42, 'E': 1.51, 'L': 1.21, 'M': 1.45, 'Q': 1.11,
            'R': 0.98, 'K': 1.16, 'H': 1.00, 'V': 1.06, 'I': 1.08,
            'Y': 0.69, 'F': 1.13, 'W': 1.08, 'T': 0.83, 'S': 0.77,
            'C': 0.70, 'P': 0.57, 'N': 0.67, 'D': 1.01, 'G': 0.57
        }
        
        sheet_prop = {
            'V': 1.70, 'I': 1.60, 'Y': 1.47, 'F': 1.38, 'W': 1.37,
            'L': 1.30, 'T': 1.19, 'C': 1.19, 'A': 0.83, 'R': 0.93,
            'G': 0.75, 'D': 0.54, 'K': 0.74, 'S': 0.75, 'H': 0.87,
            'Q': 1.10, 'P': 0.55, 'N': 0.89, 'E': 0.37, 'M': 1.05
        }
        
        helix_values = [helix_prop.get(aa, 1.0) for aa in sequence]
        sheet_values = [sheet_prop.get(aa, 1.0) for aa in sequence]
        
        return [
            np.mean(helix_values),
            np.mean(sheet_values),
            np.mean(helix_values) / np.mean(sheet_values)  # Helix/sheet ratio
        ]
    
    def _calculate_positional_features(self, sequence):
        """Calculate position-dependent features."""
        center = len(sequence) // 2
        
        # Distance from center for each residue type
        features = []
        
        for aa in 'ACDEFGHIKLMNPQRSTVWY':
            positions = [i for i, residue in enumerate(sequence) if residue == aa]
            if positions:
                avg_distance = np.mean([abs(pos - center) for pos in positions])
                features.append(avg_distance / center)  # Normalized
            else:
                features.append(0)
        
        return features
    
    def _calculate_conservation_features(self, sequence):
        """Calculate conservation-based features (placeholder)."""
        # In a real implementation, this would use MSA data
        # For now, return dummy features
        return [0.5, 0.3, 0.7]  # Placeholder conservation scores

# Usage
extractor = AdvancedFeatureExtractor()
features = extractor.extract_features(sequences)
```

### Feature Selection

```python
from sklearn.feature_selection import SelectKBest, f_classif, RFE
from sklearn.ensemble import RandomForestClassifier

def select_features(X_train, y_train, method='univariate', k=100):
    """Select most informative features."""
    
    if method == 'univariate':
        # Univariate feature selection
        selector = SelectKBest(score_func=f_classif, k=k)
        X_selected = selector.fit_transform(X_train, y_train)
        selected_features = selector.get_support()
        
    elif method == 'rfe':
        # Recursive feature elimination
        estimator = RandomForestClassifier(n_estimators=50, random_state=42)
        selector = RFE(estimator, n_features_to_select=k)
        X_selected = selector.fit_transform(X_train, y_train)
        selected_features = selector.support_
    
    elif method == 'importance':
        # Feature importance based selection
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        
        importance_scores = rf.feature_importances_
        top_indices = np.argsort(importance_scores)[-k:]
        
        selected_features = np.zeros(X_train.shape[1], dtype=bool)
        selected_features[top_indices] = True
        X_selected = X_train[:, selected_features]
    
    return X_selected, selected_features

# Usage
X_selected, feature_mask = select_features(
    X_train, y_train, method='importance', k=50
)
```

## Advanced Evaluation

### Statistical Significance Testing

```python
from scipy import stats
import numpy as np

def compare_models_statistical(results1, results2, metric='accuracy'):
    """Compare two models with statistical significance testing."""
    
    scores1 = np.array([r[metric] for r in results1])
    scores2 = np.array([r[metric] for r in results2])
    
    # Paired t-test
    t_stat, t_pvalue = stats.ttest_rel(scores1, scores2)
    
    # Wilcoxon signed-rank test
    w_stat, w_pvalue = stats.wilcoxon(scores1, scores2)
    
    # Effect size (Cohen's d)
    pooled_std = np.sqrt((np.var(scores1) + np.var(scores2)) / 2)
    cohens_d = (np.mean(scores1) - np.mean(scores2)) / pooled_std
    
    results = {
        'mean_diff': np.mean(scores1) - np.mean(scores2),
        'std_diff': np.std(scores1 - scores2),
        't_statistic': t_stat,
        't_pvalue': t_pvalue,
        'wilcoxon_statistic': w_stat,
        'wilcoxon_pvalue': w_pvalue,
        'cohens_d': cohens_d,
        'significant_t': t_pvalue < 0.05,
        'significant_w': w_pvalue < 0.05
    }
    
    return results

# Usage
xgb_results = [{'accuracy': 0.85}, {'accuracy': 0.87}, {'accuracy': 0.86}]
transformer_results = [{'accuracy': 0.88}, {'accuracy': 0.90}, {'accuracy': 0.89}]

comparison = compare_models_statistical(xgb_results, transformer_results)
print(f"Mean difference: {comparison['mean_diff']:.4f}")
print(f"p-value (t-test): {comparison['t_pvalue']:.4f}")
print(f"Significant: {comparison['significant_t']}")
```

### Cross-Validation with Custom Splitting

```python
from sklearn.model_selection import BaseCrossValidator
import pandas as pd

class ProteinGroupKFold(BaseCrossValidator):
    """Cross-validation with protein-level splitting."""
    
    def __init__(self, n_splits=5, shuffle=True, random_state=None):
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state
    
    def split(self, X, y=None, groups=None):
        """Generate train/test splits at protein level."""
        
        if groups is None:
            raise ValueError("groups parameter is required")
        
        unique_groups = np.unique(groups)
        n_groups = len(unique_groups)
        
        if self.shuffle:
            rng = np.random.RandomState(self.random_state)
            unique_groups = rng.permutation(unique_groups)
        
        group_fold_size = n_groups // self.n_splits
        
        for i in range(self.n_splits):
            start_idx = i * group_fold_size
            end_idx = (i + 1) * group_fold_size if i < self.n_splits - 1 else n_groups
            
            test_groups = set(unique_groups[start_idx:end_idx])
            train_groups = set(unique_groups) - test_groups
            
            train_indices = [idx for idx, group in enumerate(groups) 
                           if group in train_groups]
            test_indices = [idx for idx, group in enumerate(groups) 
                          if group in test_groups]
            
            yield np.array(train_indices), np.array(test_indices)
    
    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_splits

# Usage
def protein_level_cv(X, y, protein_ids, model_class, config, n_splits=5):
    """Perform protein-level cross-validation."""
    
    cv = ProteinGroupKFold(n_splits=n_splits, random_state=42)
    
    results = []
    for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, protein_ids)):
        print(f"Fold {fold + 1}/{n_splits}")
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = model_class(config)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        fold_results = {
            'fold': fold,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob),
            'n_train': len(train_idx),
            'n_test': len(test_idx)
        }
        
        results.append(fold_results)
    
    return results

# Usage
cv_results = protein_level_cv(X, y, protein_ids, XGBoostModel, config)
```

## Performance Optimization

### Memory Management

```python
from utils.memory import MemoryManager, MemoryProfiler

def train_with_memory_optimization(model, train_data, val_data):
    """Train model with memory optimization."""
    
    memory_manager = MemoryManager()
    
    # Check memory requirements
    estimated_memory = estimate_training_memory(model, train_data)
    if not memory_manager.check_available_memory(estimated_memory):
        # Reduce batch size or use gradient accumulation
        original_batch_size = model.config.get('batch_size', 32)
        optimized_batch_size = memory_manager.estimate_optimal_batch_size(
            model_memory=estimated_memory,
            sample_memory=0.1  # GB per sample
        )
        model.config['batch_size'] = min(original_batch_size, optimized_batch_size)
    
    # Profile memory usage during training
    with MemoryProfiler("Training") as profiler:
        model.fit(train_data, val_data)
    
    # Clean up memory
    memory_manager.clear_memory()
    
    return model

def batch_prediction_with_memory_limit(model, X, max_memory_gb=2.0):
    """Make predictions in batches to limit memory usage."""
    
    memory_manager = MemoryManager()
    
    # Estimate batch size
    sample_memory_mb = estimate_sample_memory(X[0])
    batch_size = memory_manager.estimate_batch_size(
        model_memory_gb=0.5,  # Estimated model memory
        sample_size_mb=sample_memory_mb,
        safety_factor=0.8
    )
    
    predictions = []
    
    for i in range(0, len(X), batch_size):
        batch = X[i:i+batch_size]
        
        with MemoryProfiler(f"Batch {i//batch_size + 1}"):
            batch_pred = model.predict_proba(batch)
            predictions.extend(batch_pred)
        
        # Clear cache between batches
        memory_manager.clear_gpu_cache()
    
    return np.array(predictions)
```

### Parallel Processing

```python
from multiprocessing import Pool
from functools import partial
import joblib

def parallel_cross_validation(X, y, model_class, config, cv_splits, n_jobs=-1):
    """Parallel cross-validation execution."""
    
    def train_fold(fold_data):
        fold_idx, (train_idx, test_idx) = fold_data
        
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Train model
        model = model_class(config)
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        y_prob = model.predict_proba(X_test)[:, 1]
        
        return {
            'fold': fold_idx,
            'accuracy': accuracy_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_prob)
        }
    
    # Parallel execution
    fold_data = list(enumerate(cv_splits))
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(train_fold)(fold) for fold in fold_data
    )
    
    return results

def parallel_hyperparameter_search(param_grid, train_data, val_data, n_jobs=-1):
    """Parallel hyperparameter search."""
    
    def evaluate_params(params):
        model = XGBoostModel(params)
        model.fit(train_data[0], train_data[1], val_data[0], val_data[1])
        
        y_pred = model.predict_proba(val_data[0])[:, 1]
        score = f1_score(val_data[1], (y_pred > 0.5).astype(int))
        
        return {'params': params, 'score': score}
    
    # Generate parameter combinations
    param_combinations = list(product(*param_grid.values()))
    param_dicts = [dict(zip(param_grid.keys(), combo)) 
                   for combo in param_combinations]
    
    # Parallel evaluation
    results = joblib.Parallel(n_jobs=n_jobs)(
        joblib.delayed(evaluate_params)(params) for params in param_dicts
    )
    
    return results
```

## Custom Callbacks and Monitoring

### Custom Training Callbacks

```python
from training.callbacks import Callback
import matplotlib.pyplot as plt

class CustomVisualizationCallback(Callback):
    """Custom callback for real-time visualization."""
    
    def __init__(self, plot_frequency=10):
        super().__init__()
        self.plot_frequency = plot_frequency
        self.train_losses = []
        self.val_losses = []
        self.train_metrics = []
        self.val_metrics = []
    
    def on_epoch_end(self, epoch, logs=None):
        """Called at the end of each epoch."""
        logs = logs or {}
        
        # Store metrics
        self.train_losses.append(logs.get('train_loss', 0))
        self.val_losses.append(logs.get('val_loss', 0))
        self.train_metrics.append(logs.get('train_f1', 0))
        self.val_metrics.append(logs.get('val_f1', 0))
        
        # Plot every N epochs
        if epoch % self.plot_frequency == 0:
            self._plot_training_progress()
    
    def _plot_training_progress(self):
        """Plot training progress."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        
        # Loss plot
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Val Loss')
        ax1.set_title('Training Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        
        # Metric plot
        ax2.plot(self.train_metrics, label='Train F1')
        ax2.plot(self.val_metrics, label='Val F1')
        ax2.set_title('F1 Score')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('F1 Score')
        ax2.legend()
        
        plt.tight_layout()
        plt.savefig('training_progress.png')
        plt.close()

class AdaptiveLearningRateCallback(Callback):
    """Adaptive learning rate based on validation performance."""
    
    def __init__(self, patience=5, factor=0.5, min_lr=1e-7):
        super().__init__()
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.best_score = -np.inf
        self.wait = 0
    
    def on_epoch_end(self, epoch, logs=None):
        """Adjust learning rate based on validation performance."""
        logs = logs or {}
        current_score = logs.get('val_f1', 0)
        
        if current_score > self.best_score:
            self.best_score = current_score
            self.wait = 0
        else:
            self.wait += 1
            
            if self.wait >= self.patience:
                old_lr = self.model.get_learning_rate()
                new_lr = max(old_lr * self.factor, self.min_lr)
                
                if new_lr < old_lr:
                    self.model.set_learning_rate(new_lr)
                    print(f"Reduced learning rate to {new_lr}")
                    self.wait = 0

# Usage
trainer = Trainer(config)
trainer.add_callback(CustomVisualizationCallback(plot_frequency=5))
trainer.add_callback(AdaptiveLearningRateCallback(patience=10))
trainer.fit(model, train_data, val_data)
```

This advanced usage guide covers programmatic API usage, custom experiments, hyperparameter optimization, advanced feature engineering, statistical evaluation, and performance optimization techniques. These examples show how to extend and customize the framework for specific research needs.