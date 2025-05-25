#!/usr/bin/env python3
"""Analysis script for phosphorylation prediction experiments."""

import argparse
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import load_config
from evaluation import ModelAnalyzer, plot_confusion_matrix, plot_roc_curves, plot_feature_importance
from utils import get_logger, load_experiment_checkpoint

def main():
    parser = argparse.ArgumentParser(description="Analyze phosphorylation prediction results")
    parser.add_argument("--experiment-dir", type=str, required=True,
                       help="Path to experiment directory")
    parser.add_argument("--analysis-type", type=str,
                       choices=["metrics", "comparison", "feature_importance", "error_analysis", "all"],
                       default="all", help="Type of analysis to perform")
    parser.add_argument("--output-dir", type=str,
                       help="Output directory for analysis results")
    parser.add_argument("--compare-experiments", type=str, nargs="+",
                       help="Additional experiment directories for comparison")
    parser.add_argument("--format", type=str, choices=["png", "pdf", "svg"],
                       default="png", help="Output format for plots")
    parser.add_argument("--dpi", type=int, default=300,
                       help="DPI for output plots")
    
    args = parser.parse_args()
    
    # Setup
    logger = get_logger(__name__)
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.experiment_dir) / "analysis"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Analyzing experiment: {args.experiment_dir}")
    logger.info(f"Output directory: {output_dir}")
    
    try:
        # Load experiment data
        experiment_dir = Path(args.experiment_dir)
        
        # Look for experiment checkpoint
        checkpoint_files = list(experiment_dir.rglob("experiment_checkpoint.json"))
        if not checkpoint_files:
            raise FileNotFoundError("No experiment checkpoint found")
        
        checkpoint_path = checkpoint_files[0]
        experiment_data = load_experiment_checkpoint(checkpoint_path)
        
        config = experiment_data['config']
        metrics = experiment_data['metrics']
        models = experiment_data.get('models', {})
        
        logger.info(f"Loaded experiment with {len(models)} models")
        
        # Initialize analyzer
        analyzer = ModelAnalyzer()
        
        # Load additional experiments for comparison
        comparison_data = []
        if args.compare_experiments:
            for comp_dir in args.compare_experiments:
                comp_path = Path(comp_dir)
                comp_checkpoints = list(comp_path.rglob("experiment_checkpoint.json"))
                if comp_checkpoints:
                    comp_data = load_experiment_checkpoint(comp_checkpoints[0])
                    comparison_data.append({
                        'name': comp_path.name,
                        'data': comp_data
                    })
                    logger.info(f"Loaded comparison experiment: {comp_path.name}")
        
        # Perform analysis
        if args.analysis_type in ["metrics", "all"]:
            logger.info("Generating metrics analysis...")
            
            # Create metrics summary
            metrics_df = pd.DataFrame([metrics])
            metrics_df['experiment'] = experiment_dir.name
            
            # Add comparison metrics
            for comp in comparison_data:
                comp_metrics = pd.DataFrame([comp['data']['metrics']])
                comp_metrics['experiment'] = comp['name']
                metrics_df = pd.concat([metrics_df, comp_metrics], ignore_index=True)
            
            # Save metrics table
            metrics_path = output_dir / "metrics_summary.csv"
            metrics_df.to_csv(metrics_path, index=False)
            logger.info(f"Metrics summary saved to {metrics_path}")
            
            # Create metrics comparison plot
            if len(metrics_df) > 1:
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score']
                
                for i, metric in enumerate(metrics_to_plot):
                    ax = axes[i//2, i%2]
                    if metric in metrics_df.columns:
                        bars = ax.bar(metrics_df['experiment'], metrics_df[metric])
                        ax.set_title(f'{metric.title()}')
                        ax.set_ylabel('Score')
                        ax.tick_params(axis='x', rotation=45)
                        
                        # Add value labels on bars
                        for bar in bars:
                            height = bar.get_height()
                            ax.annotate(f'{height:.3f}',
                                      xy=(bar.get_x() + bar.get_width()/2, height),
                                      xytext=(0, 3),
                                      textcoords="offset points",
                                      ha='center', va='bottom')
                
                plt.tight_layout()
                plt.savefig(output_dir / f"metrics_comparison.{args.format}", 
                           dpi=args.dpi, bbox_inches='tight')
                plt.close()
        
        if args.analysis_type in ["feature_importance", "all"]:
            logger.info("Analyzing feature importance...")
            
            # Check if we have XGBoost models with feature importance
            for model_name, model in models.items():
                if hasattr(model, 'feature_importances_'):
                    importance_plot_path = output_dir / f"feature_importance_{model_name}.{args.format}"
                    plot_feature_importance(
                        model, 
                        save_path=importance_plot_path,
                        top_k=20
                    )
                    logger.info(f"Feature importance plot saved to {importance_plot_path}")
        
        if args.analysis_type in ["error_analysis", "all"]:
            logger.info("Performing error analysis...")
            
            # Look for prediction files
            pred_files = list(experiment_dir.rglob("*predictions*.csv"))
            if pred_files:
                for pred_file in pred_files:
                    df = pd.read_csv(pred_file)
                    
                    if 'y_true' in df.columns and 'y_pred' in df.columns:
                        # Confusion matrix
                        cm_path = output_dir / f"confusion_matrix_{pred_file.stem}.{args.format}"
                        plot_confusion_matrix(
                            df['y_true'], 
                            df['y_pred'],
                            save_path=cm_path
                        )
                        
                        # Error analysis by sequence length
                        if 'sequence_length' in df.columns:
                            error_analysis = analyzer.analyze_errors_by_length(
                                df['y_true'], 
                                df['y_pred'], 
                                df['sequence_length']
                            )
                            
                            error_path = output_dir / f"error_analysis_{pred_file.stem}.csv"
                            error_analysis.to_csv(error_path, index=False)
                            logger.info(f"Error analysis saved to {error_path}")
        
        if args.analysis_type in ["comparison", "all"] and comparison_data:
            logger.info("Generating model comparison...")
            
            # ROC curve comparison
            all_predictions = []
            
            # Load predictions from all experiments
            for comp in [{'name': experiment_dir.name, 'data': experiment_data}] + comparison_data:
                comp_dir = experiment_dir if comp['name'] == experiment_dir.name else Path(args.compare_experiments[0]).parent / comp['name']
                pred_files = list(comp_dir.rglob("*predictions*.csv"))
                
                for pred_file in pred_files:
                    df = pd.read_csv(pred_file)
                    if 'y_true' in df.columns and 'y_prob' in df.columns:
                        all_predictions.append({
                            'name': f"{comp['name']}_{pred_file.stem}",
                            'y_true': df['y_true'],
                            'y_prob': df['y_prob']
                        })
            
            if all_predictions:
                roc_path = output_dir / f"roc_comparison.{args.format}"
                plot_roc_curves(all_predictions, save_path=roc_path)
                logger.info(f"ROC comparison saved to {roc_path}")
        
        # Generate analysis report
        report_path = output_dir / "analysis_report.txt"
        with open(report_path, 'w') as f:
            f.write("Phosphorylation Prediction Analysis Report\n")
            f.write("=" * 50 + "\n\n")
            
            f.write(f"Experiment: {experiment_dir.name}\n")
            f.write(f"Analysis Date: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            
            f.write("Configuration:\n")
            f.write("-" * 20 + "\n")
            for key, value in config.items():
                f.write(f"{key}: {value}\n")
            f.write("\n")
            
            f.write("Metrics:\n")
            f.write("-" * 20 + "\n")
            for key, value in metrics.items():
                if isinstance(value, float):
                    f.write(f"{key}: {value:.4f}\n")
                else:
                    f.write(f"{key}: {value}\n")
            f.write("\n")
            
            if comparison_data:
                f.write("Comparison Results:\n")
                f.write("-" * 20 + "\n")
                for comp in comparison_data:
                    f.write(f"\n{comp['name']}:\n")
                    comp_metrics = comp['data']['metrics']
                    for key, value in comp_metrics.items():
                        if isinstance(value, float):
                            f.write(f"  {key}: {value:.4f}\n")
                        else:
                            f.write(f"  {key}: {value}\n")
        
        logger.info(f"Analysis report saved to {report_path}")
        logger.info("Analysis completed successfully!")
    
    except Exception as e:
        logger.error(f"Analysis failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()