"""Visualization tools for phosphorylation prediction evaluation."""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Any, Union, Tuple
import os
from datetime import datetime
import logging

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def plot_roc_curves(results_dict: Dict[str, Dict[str, Any]], 
                   save_path: Optional[str] = None,
                   figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot ROC curves for multiple models.
    
    Args:
        results_dict: Dictionary with model results containing y_true, y_proba
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import roc_curve, auc
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, results in results_dict.items():
        y_true = results['y_true']
        y_proba = results['y_proba']
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)
        
        ax.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    ax.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title('Receiver Operating Characteristic (ROC) Curves', fontsize=14)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_precision_recall_curves(results_dict: Dict[str, Dict[str, Any]], 
                                save_path: Optional[str] = None,
                                figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot Precision-Recall curves for multiple models.
    
    Args:
        results_dict: Dictionary with model results containing y_true, y_proba
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import precision_recall_curve, average_precision_score
    
    fig, ax = plt.subplots(figsize=figsize)
    
    for model_name, results in results_dict.items():
        y_true = results['y_true']
        y_proba = results['y_proba']
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        avg_precision = average_precision_score(y_true, y_proba)
        
        ax.plot(recall, precision, lw=2, label=f'{model_name} (AP = {avg_precision:.3f})')
    
    # Plot baseline (random classifier)
    baseline = np.mean(results_dict[list(results_dict.keys())[0]]['y_true'])
    ax.axhline(y=baseline, color='k', linestyle='--', lw=2, label=f'Random Classifier (AP = {baseline:.3f})')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('Recall', fontsize=12)
    ax.set_ylabel('Precision', fontsize=12)
    ax.set_title('Precision-Recall Curves', fontsize=14)
    ax.legend(loc="lower left")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_confusion_matrices(results_dict: Dict[str, Dict[str, Any]], 
                          save_path: Optional[str] = None,
                          figsize: Optional[Tuple[int, int]] = None) -> plt.Figure:
    """
    Plot confusion matrices for multiple models.
    
    Args:
        results_dict: Dictionary with model results containing y_true, y_pred
        save_path: Path to save the plot (optional)
        figsize: Figure size (auto-calculated if None)
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import confusion_matrix
    
    n_models = len(results_dict)
    cols = min(3, n_models)
    rows = (n_models + cols - 1) // cols
    
    if figsize is None:
        figsize = (5 * cols, 4 * rows)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_models == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    for idx, (model_name, results) in enumerate(results_dict.items()):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        y_true = results['y_true']
        y_pred = results['y_pred']
        
        cm = confusion_matrix(y_true, y_pred)
        
        # Plot heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Negative', 'Positive'],
                   yticklabels=['Negative', 'Positive'],
                   ax=ax)
        
        ax.set_title(f'{model_name}', fontsize=12)
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
    
    # Hide unused subplots
    for idx in range(n_models, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_feature_importance(importance_dict: Dict[str, float], 
                          top_n: int = 30,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (12, 10)) -> plt.Figure:
    """
    Plot feature importance.
    
    Args:
        importance_dict: Dictionary of feature names and importance scores
        top_n: Number of top features to plot
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    # Sort by importance
    sorted_features = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)[:top_n]
    
    features, importances = zip(*sorted_features)
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Create horizontal bar plot
    y_pos = np.arange(len(features))
    bars = ax.barh(y_pos, importances, color='steelblue', alpha=0.7)
    
    ax.set_yticks(y_pos)
    ax.set_yticklabels(features)
    ax.invert_yaxis()  # Top feature at the top
    ax.set_xlabel('Importance Score', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importances', fontsize=14)
    
    # Add value labels on bars
    for i, (bar, importance) in enumerate(zip(bars, importances)):
        ax.text(bar.get_width() + max(importances) * 0.01, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', ha='left', va='center', fontsize=8)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_training_curves(training_history: Dict[str, List[Dict[str, float]]], 
                        save_path: Optional[str] = None,
                        figsize: Tuple[int, int] = (15, 10)) -> plt.Figure:
    """
    Plot training and validation curves.
    
    Args:
        training_history: Dictionary with 'train' and 'val' keys containing metrics per epoch
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if not training_history or 'train' not in training_history:
        raise ValueError("training_history must contain 'train' key with metrics")
    
    train_metrics = training_history['train']
    val_metrics = training_history.get('val', [])
    
    if not train_metrics:
        raise ValueError("No training metrics found")
    
    # Get all available metrics
    metric_names = list(train_metrics[0].keys())
    metric_names = [m for m in metric_names if not m.startswith('_')]  # Exclude private metrics
    
    # Create subplots
    n_metrics = len(metric_names)
    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    elif rows == 1:
        axes = axes.reshape(1, -1)
    
    epochs = range(1, len(train_metrics) + 1)
    
    for idx, metric in enumerate(metric_names):
        row = idx // cols
        col = idx % cols
        ax = axes[row, col] if rows > 1 else axes[col]
        
        # Plot training curve
        train_values = [epoch_metrics[metric] for epoch_metrics in train_metrics 
                       if metric in epoch_metrics]
        ax.plot(epochs[:len(train_values)], train_values, 'b-', label='Training', linewidth=2)
        
        # Plot validation curve if available
        if val_metrics:
            val_values = [epoch_metrics[metric] for epoch_metrics in val_metrics 
                         if metric in epoch_metrics]
            if val_values:
                ax.plot(epochs[:len(val_values)], val_values, 'r-', label='Validation', linewidth=2)
        
        ax.set_xlabel('Epoch', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=10)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_metrics, rows * cols):
        row = idx // cols
        col = idx % cols
        if rows > 1:
            axes[row, col].set_visible(False)
        else:
            axes[col].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_calibration_curve(y_true: np.ndarray, y_proba: np.ndarray,
                          n_bins: int = 10, save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (10, 8)) -> plt.Figure:
    """
    Plot calibration curve (reliability diagram).
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.calibration import calibration_curve
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Plot calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_true, y_proba, n_bins=n_bins, strategy='uniform'
    )
    
    ax1.plot(mean_predicted_value, fraction_of_positives, "s-", label="Model")
    ax1.plot([0, 1], [0, 1], "k--", label="Perfectly calibrated")
    ax1.set_xlabel('Mean Predicted Probability')
    ax1.set_ylabel('Fraction of Positives')
    ax1.set_title('Calibration Curve')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot histogram of predicted probabilities
    ax2.hist(y_proba, bins=50, alpha=0.7, density=True, color='steelblue')
    ax2.set_xlabel('Predicted Probability')
    ax2.set_ylabel('Density')
    ax2.set_title('Distribution of Predicted Probabilities')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                          save_path: Optional[str] = None,
                          figsize: Tuple[int, int] = (15, 5)) -> plt.Figure:
    """
    Plot threshold analysis showing precision, recall, and F1 vs threshold.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    from sklearn.metrics import precision_recall_curve
    from ..evaluation.metrics import calculate_threshold_metrics
    
    # Calculate metrics for different thresholds
    threshold_df = calculate_threshold_metrics(y_true, y_proba)
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot Precision vs Threshold
    axes[0].plot(threshold_df['threshold'], threshold_df['precision'], 'b-', linewidth=2)
    axes[0].set_xlabel('Threshold')
    axes[0].set_ylabel('Precision')
    axes[0].set_title('Precision vs Threshold')
    axes[0].grid(True, alpha=0.3)
    
    # Plot Recall vs Threshold
    axes[1].plot(threshold_df['threshold'], threshold_df['recall'], 'r-', linewidth=2)
    axes[1].set_xlabel('Threshold')
    axes[1].set_ylabel('Recall')
    axes[1].set_title('Recall vs Threshold')
    axes[1].grid(True, alpha=0.3)
    
    # Plot F1 vs Threshold
    axes[2].plot(threshold_df['threshold'], threshold_df['f1'], 'g-', linewidth=2)
    
    # Mark optimal F1 threshold
    optimal_idx = threshold_df['f1'].idxmax()
    optimal_threshold = threshold_df.loc[optimal_idx, 'threshold']
    optimal_f1 = threshold_df.loc[optimal_idx, 'f1']
    
    axes[2].axvline(x=optimal_threshold, color='orange', linestyle='--', 
                   label=f'Optimal (F1={optimal_f1:.3f})')
    axes[2].set_xlabel('Threshold')
    axes[2].set_ylabel('F1 Score')
    axes[2].set_title('F1 Score vs Threshold')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def plot_model_comparison(comparison_df: pd.DataFrame, 
                         metrics_to_plot: Optional[List[str]] = None,
                         save_path: Optional[str] = None,
                         figsize: Tuple[int, int] = (14, 10)) -> plt.Figure:
    """
    Plot model comparison across multiple metrics.
    
    Args:
        comparison_df: DataFrame with model comparison results
        metrics_to_plot: List of metrics to plot (optional)
        save_path: Path to save the plot (optional)
        figsize: Figure size
        
    Returns:
        Matplotlib figure
    """
    if metrics_to_plot is None:
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    # Filter available metrics
    available_metrics = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.ravel()
    
    # Bar plot of all metrics
    ax = axes[0]
    x = np.arange(len(comparison_df))
    width = 0.15
    
    for i, metric in enumerate(available_metrics):
        ax.bar(x + i * width, comparison_df[metric], width, label=metric, alpha=0.8)
    
    ax.set_xlabel('Models')
    ax.set_ylabel('Score')
    ax.set_title('Model Performance Comparison')
    ax.set_xticks(x + width * (len(available_metrics) - 1) / 2)
    ax.set_xticklabels(comparison_df['model'], rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Radar chart
    if len(available_metrics) >= 3:
        ax = axes[1]
        
        # Number of metrics
        num_metrics = len(available_metrics)
        angles = np.linspace(0, 2 * np.pi, num_metrics, endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        for _, row in comparison_df.iterrows():
            values = [row[metric] for metric in available_metrics]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=row['model'])
            ax.fill(angles, values, alpha=0.25)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(available_metrics)
        ax.set_ylim(0, 1)
        ax.set_title('Model Performance Radar Chart')
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        # Make it a polar plot
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
    
    # Heatmap of metrics
    ax = axes[2]
    heatmap_data = comparison_df.set_index('model')[available_metrics]
    
    sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='viridis', ax=ax, cbar_kws={'shrink': 0.8})
    ax.set_title('Model Performance Heatmap')
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Models')
    
    # Ranking plot
    ax = axes[3]
    ranking_cols = [col for col in comparison_df.columns if col.endswith('_rank')]
    
    if ranking_cols:
        ranking_data = comparison_df.set_index('model')[ranking_cols]
        ranking_data.columns = [col.replace('_rank', '') for col in ranking_data.columns]
        
        # Plot ranking (lower rank is better)
        for model in ranking_data.index:
            ax.plot(ranking_data.columns, ranking_data.loc[model], 'o-', 
                   label=model, linewidth=2, markersize=8)
        
        ax.set_xlabel('Metrics')
        ax.set_ylabel('Rank (lower is better)')
        ax.set_title('Model Rankings by Metric')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.invert_yaxis()  # Lower ranks at the top
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig


def create_dashboard(results: Dict[str, Any], save_path: str) -> None:
    """
    Create comprehensive HTML dashboard with all visualizations.
    
    Args:
        results: Dictionary containing all analysis results
        save_path: Path to save the HTML dashboard
    """
    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Phosphorylation Prediction Results Dashboard</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; }}
            .header {{ text-align: center; color: #333; }}
            .section {{ margin: 20px 0; }}
            .metrics-table {{ border-collapse: collapse; width: 100%; }}
            .metrics-table th, .metrics-table td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .metrics-table th {{ background-color: #f2f2f2; }}
            .plot-container {{ text-align: center; margin: 20px 0; }}
            .plot-container img {{ max-width: 100%; height: auto; }}
        </style>
    </head>
    <body>
        <div class="header">
            <h1>Phosphorylation Site Prediction Results</h1>
            <p>Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        </div>
        
        <div class="section">
            <h2>Model Performance Summary</h2>
            <!-- Add performance tables and summaries here -->
        </div>
        
        <div class="section">
            <h2>ROC Curves</h2>
            <div class="plot-container">
                <img src="roc_curves.png" alt="ROC Curves">
            </div>
        </div>
        
        <div class="section">
            <h2>Precision-Recall Curves</h2>
            <div class="plot-container">
                <img src="pr_curves.png" alt="Precision-Recall Curves">
            </div>
        </div>
        
        <div class="section">
            <h2>Confusion Matrices</h2>
            <div class="plot-container">
                <img src="confusion_matrices.png" alt="Confusion Matrices">
            </div>
        </div>
        
        <div class="section">
            <h2>Feature Importance</h2>
            <div class="plot-container">
                <img src="feature_importance.png" alt="Feature Importance">
            </div>
        </div>
        
        <div class="section">
            <h2>Training Curves</h2>
            <div class="plot-container">
                <img src="training_curves.png" alt="Training Curves">
            </div>
        </div>
        
    </body>
    </html>
    """
    
    # Create directory and save HTML
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        f.write(html_content)
    
    print(f"Dashboard saved to {save_path}")


def plot_results(results: Dict[str, Any], output_dir: str) -> None:
    """
    Generate all plots and save them to output directory.
    
    Args:
        results: Dictionary containing all results and data
        output_dir: Directory to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger = logging.getLogger(__name__)
    logger.info(f"Generating plots in {output_dir}")
    
    try:
        # ROC curves
        if 'model_results' in results:
            plot_roc_curves(results['model_results'], 
                           save_path=os.path.join(output_dir, 'roc_curves.png'))
            
            plot_precision_recall_curves(results['model_results'],
                                        save_path=os.path.join(output_dir, 'pr_curves.png'))
            
            plot_confusion_matrices(results['model_results'],
                                   save_path=os.path.join(output_dir, 'confusion_matrices.png'))
        
        # Feature importance
        if 'feature_importance' in results:
            plot_feature_importance(results['feature_importance'],
                                   save_path=os.path.join(output_dir, 'feature_importance.png'))
        
        # Training curves
        if 'training_history' in results:
            plot_training_curves(results['training_history'],
                                save_path=os.path.join(output_dir, 'training_curves.png'))
        
        # Model comparison
        if 'comparison_df' in results:
            plot_model_comparison(results['comparison_df'],
                                save_path=os.path.join(output_dir, 'model_comparison.png'))
        
        # Calibration curves
        if 'calibration_data' in results:
            for model_name, data in results['calibration_data'].items():
                plot_calibration_curve(data['y_true'], data['y_proba'],
                                     save_path=os.path.join(output_dir, f'calibration_{model_name}.png'))
        
        logger.info("All plots generated successfully")
        
    except Exception as e:
        logger.error(f"Error generating plots: {e}")
        raise