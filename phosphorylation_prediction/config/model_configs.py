"""Model configuration classes for phosphorylation prediction."""

from dataclasses import dataclass
from typing import List, Optional, Union, Dict, Any


@dataclass
class BaseConfig:
    """Base configuration with common parameters."""
    seed: int = 42
    device: str = "cuda"
    num_workers: int = 4


@dataclass
class DataConfig(BaseConfig):
    """Data processing configuration."""
    sequence_col: str = "Sequence"
    position_col: str = "Position"
    target_col: str = "target"
    protein_id_col: str = "Header"
    window_size: int = 10
    max_sequence_length: int = 5000
    balance_classes: bool = True
    augmentation: bool = False


@dataclass
class TrainingConfig(BaseConfig):
    """Training configuration."""
    batch_size: int = 32
    num_epochs: int = 10
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    gradient_accumulation_steps: int = 4
    fp16: bool = True
    early_stopping_patience: int = 5


@dataclass
class XGBoostConfig(BaseConfig):
    """XGBoost model configuration."""
    n_estimators: int = 1000
    max_depth: int = 6
    learning_rate: float = 0.1
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    early_stopping_rounds: int = 50
    eval_metric: List[str] = None
    use_gpu: bool = False
    
    def __post_init__(self):
        if self.eval_metric is None:
            self.eval_metric = ["logloss", "auc"]


@dataclass
class TransformerConfig(BaseConfig):
    """Transformer model configuration."""
    model_name: str = "facebook/esm2_t6_8M_UR50D"
    max_length: int = 64
    dropout_rate: float = 0.3
    window_context: int = 3
    warmup_steps: int = 500


@dataclass
class EnsembleConfig(BaseConfig):
    """Ensemble method configuration."""
    voting_strategy: str = "soft"  # soft or hard
    voting_weights: Union[str, List[float]] = "optimize"  # optimize, equal, or list
    stacking_meta_learner: str = "logistic_regression"
    stacking_use_probas: bool = True
    stacking_cv_predictions: bool = True
    blending_ratio: float = 0.2
    dynamic_k_neighbors: int = 5
    dynamic_similarity_metric: str = "cosine"


@dataclass
class WandbConfig:
    """Weights & Biases configuration."""
    entity: Optional[str] = None
    project: str = "phospho-prediction"
    tags: List[str] = None
    log_frequency: int = 50
    log_predictions: bool = True
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = ["baseline"]


@dataclass
class FeatureConfig:
    """Feature extraction configuration."""
    use_aac: bool = True
    use_dpc: bool = True
    use_tpc: bool = True
    use_binary: bool = True
    use_physicochemical: bool = True


def create_config_from_dict(config_dict: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create configuration objects from dictionary.
    
    Args:
        config_dict: Configuration dictionary
        
    Returns:
        Dictionary with configuration objects
    """
    configs = {}
    
    # Create data config
    if 'data' in config_dict:
        configs['data'] = DataConfig(**config_dict['data'])
    
    # Create training config
    if 'training' in config_dict:
        configs['training'] = TrainingConfig(**config_dict['training'])
    
    # Create XGBoost config
    if 'xgboost' in config_dict:
        configs['xgboost'] = XGBoostConfig(**config_dict['xgboost'])
    
    # Create Transformer config
    if 'transformer' in config_dict:
        configs['transformer'] = TransformerConfig(**config_dict['transformer'])
    
    # Create Ensemble config
    if 'ensemble' in config_dict:
        configs['ensemble'] = EnsembleConfig(**config_dict['ensemble'])
    
    # Create WandB config
    if 'wandb' in config_dict:
        configs['wandb'] = WandbConfig(**config_dict['wandb'])
    
    # Create Feature config
    if 'features' in config_dict:
        configs['features'] = FeatureConfig(**config_dict['features'])
    
    return configs