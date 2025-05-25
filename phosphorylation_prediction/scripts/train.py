#!/usr/bin/env python3
"""Main training script for phosphorylation prediction models."""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import load_config
from experiments import SingleModelExperiment, EnsembleExperiment, CrossValidationExperiment
from utils import set_seed, get_logger

def main():
    parser = argparse.ArgumentParser(description="Train phosphorylation prediction models")
    parser.add_argument("--config", type=str, required=True,
                       help="Path to configuration file")
    parser.add_argument("--experiment-type", type=str, 
                       choices=["single", "ensemble", "cross_validation"],
                       default="single", help="Type of experiment to run")
    parser.add_argument("--model-type", type=str,
                       choices=["xgboost", "transformer", "ensemble"],
                       help="Model type (for single model experiments)")
    parser.add_argument("--data-path", type=str, required=True,
                       help="Path to training data")
    parser.add_argument("--output-dir", type=str, required=True,
                       help="Output directory for results")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed for reproducibility")
    parser.add_argument("--wandb-project", type=str,
                       help="Weights & Biases project name")
    parser.add_argument("--wandb-run-name", type=str,
                       help="Weights & Biases run name")
    parser.add_argument("--resume", type=str,
                       help="Path to checkpoint to resume from")
    parser.add_argument("--debug", action="store_true",
                       help="Enable debug mode")
    
    args = parser.parse_args()
    
    # Setup
    set_seed(args.seed)
    logger = get_logger(__name__)
    
    if args.debug:
        logger.setLevel("DEBUG")
    
    # Load configuration
    config = load_config(args.config)
    
    # Override config with command line arguments
    if args.wandb_project:
        config['wandb']['project'] = args.wandb_project
    if args.wandb_run_name:
        config['wandb']['run_name'] = args.wandb_run_name
    
    logger.info(f"Starting {args.experiment_type} experiment")
    logger.info(f"Configuration: {args.config}")
    logger.info(f"Data path: {args.data_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    try:
        # Create and run experiment
        if args.experiment_type == "single":
            if not args.model_type:
                raise ValueError("--model-type required for single model experiments")
            
            experiment = SingleModelExperiment(
                config=config,
                output_dir=args.output_dir,
                model_type=args.model_type
            )
        
        elif args.experiment_type == "ensemble":
            experiment = EnsembleExperiment(
                config=config,
                output_dir=args.output_dir
            )
        
        elif args.experiment_type == "cross_validation":
            experiment = CrossValidationExperiment(
                config=config,
                output_dir=args.output_dir,
                model_type=args.model_type or "xgboost"
            )
        
        # Load data and run experiment
        experiment.load_data(args.data_path)
        
        if args.resume:
            experiment.resume_from_checkpoint(args.resume)
        
        results = experiment.run()
        
        logger.info("Experiment completed successfully!")
        logger.info(f"Results saved to: {args.output_dir}")
        
        # Print summary metrics
        if 'test_metrics' in results:
            metrics = results['test_metrics']
            logger.info("Test Metrics:")
            for metric, value in metrics.items():
                if isinstance(value, float):
                    logger.info(f"  {metric}: {value:.4f}")
                else:
                    logger.info(f"  {metric}: {value}")
    
    except Exception as e:
        logger.error(f"Experiment failed: {str(e)}")
        if args.debug:
            raise
        sys.exit(1)

if __name__ == "__main__":
    main()