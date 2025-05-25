"""Configuration loading utilities for phosphorylation prediction."""

import os
import yaml
from typing import Dict, Any
from pathlib import Path


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load YAML configuration file and merge with defaults.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        Configuration dictionary
    """
    # Load default config first
    default_config_path = Path(__file__).parent / "default_config.yaml"
    with open(default_config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # If custom config provided, merge it
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            custom_config = yaml.safe_load(f)
        config = merge_configs(config, custom_config)
    
    # Validate configuration
    validate_config(config)
    
    return config


def save_config(config: Dict[str, Any], path: str) -> None:
    """
    Save configuration to YAML file for reproducibility.
    
    Args:
        config: Configuration dictionary
        path: Path to save configuration
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, indent=2)


def merge_configs(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base: Base configuration
        override: Override configuration
        
    Returns:
        Merged configuration
    """
    merged = base.copy()
    
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged


def validate_config(config: Dict[str, Any]) -> None:
    """
    Validate configuration parameters.
    
    Args:
        config: Configuration to validate
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_sections = ['experiment', 'data', 'cross_validation']
    
    for section in required_sections:
        if section not in config:
            raise ValueError(f"Missing required configuration section: {section}")
    
    # Validate experiment settings
    if 'project_name' not in config['experiment']:
        raise ValueError("experiment.project_name is required")
    
    # Validate data settings
    data_config = config['data']
    required_data_fields = ['sequence_col', 'position_col', 'target_col', 'protein_id_col']
    for field in required_data_fields:
        if field not in data_config:
            raise ValueError(f"data.{field} is required")
    
    # Validate cross-validation settings
    cv_config = config['cross_validation']
    if 'n_folds' not in cv_config:
        raise ValueError("cross_validation.n_folds is required")
    
    if cv_config['n_folds'] < 2:
        raise ValueError("cross_validation.n_folds must be at least 2")