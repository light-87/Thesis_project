"""Metrics calculation for phosphorylation prediction evaluation."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Union
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    roc_curve, precision_recall_curve, matthews_corrcoef,
    balanced_accuracy_score, classification_report
)
import logging


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                     y_proba: Optional[np.ndarray] = None,
                     pos_label: int = 1) -> Dict[str, float]:
    """
    Calculate comprehensive metrics for binary classification.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        pos_label: Label of positive class
        
    Returns:
        Dictionary of calculated metrics
    """
    metrics = {}
    
    # Basic classification metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
    metrics['precision'] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    metrics['f1'] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Confusion matrix components
    cm = confusion_matrix(y_true, y_pred)
    if cm.shape == (2, 2):
        tn, fp, fn, tp = cm.ravel()
        metrics['true_positives'] = int(tp)
        metrics['true_negatives'] = int(tn)
        metrics['false_positives'] = int(fp)
        metrics['false_negatives'] = int(fn)
        
        # Additional derived metrics
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        metrics['positive_predictive_value'] = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        metrics['negative_predictive_value'] = tn / (tn + fn) if (tn + fn) > 0 else 0.0
        metrics['false_positive_rate'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        metrics['false_negative_rate'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    
    # Probability-based metrics
    if y_proba is not None:
        metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        metrics['average_precision'] = average_precision_score(y_true, y_proba)
        
        # Calculate optimal threshold based on F1 score
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-8)
        optimal_idx = np.argmax(f1_scores)
        metrics['optimal_threshold'] = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        metrics['optimal_f1'] = f1_scores[optimal_idx]
    
    return metrics


def calculate_per_class_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Dict[str, float]]:
    """
    Calculate per-class metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with per-class metrics
    """
    report = classification_report(y_true, y_pred, output_dict=True, zero_division=0)
    
    per_class_metrics = {}
    for class_label in ['0', '1']:  # Binary classification
        if class_label in report:
            per_class_metrics[f'class_{class_label}'] = {
                'precision': report[class_label]['precision'],
                'recall': report[class_label]['recall'],
                'f1': report[class_label]['f1-score'],
                'support': report[class_label]['support']
            }
    
    # Add macro and weighted averages
    for avg_type in ['macro avg', 'weighted avg']:
        if avg_type in report:
            per_class_metrics[avg_type.replace(' ', '_')] = {
                'precision': report[avg_type]['precision'],
                'recall': report[avg_type]['recall'],
                'f1': report[avg_type]['f1-score'],
                'support': report[avg_type]['support']
            }
    
    return per_class_metrics


def calculate_confidence_calibration(y_true: np.ndarray, y_proba: np.ndarray, 
                                   n_bins: int = 10) -> Dict[str, Any]:
    """
    Calculate confidence calibration metrics.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        n_bins: Number of bins for calibration
        
    Returns:
        Dictionary with calibration metrics
    """
    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    calibration_data = []
    
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (y_proba > bin_lower) & (y_proba <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            accuracy_in_bin = y_true[in_bin].mean()
            avg_confidence_in_bin = y_proba[in_bin].mean()
            
            calibration_data.append({
                'bin_lower': bin_lower,
                'bin_upper': bin_upper,
                'bin_center': (bin_lower + bin_upper) / 2,
                'prop_in_bin': prop_in_bin,
                'accuracy_in_bin': accuracy_in_bin,
                'avg_confidence_in_bin': avg_confidence_in_bin,
                'count_in_bin': in_bin.sum()
            })
    
    # Calculate Expected Calibration Error (ECE)
    ece = 0.0
    for bin_data in calibration_data:
        ece += bin_data['prop_in_bin'] * abs(bin_data['accuracy_in_bin'] - bin_data['avg_confidence_in_bin'])
    
    # Calculate Maximum Calibration Error (MCE)
    mce = 0.0
    if calibration_data:
        mce = max(abs(bin_data['accuracy_in_bin'] - bin_data['avg_confidence_in_bin']) 
                 for bin_data in calibration_data)
    
    return {
        'expected_calibration_error': ece,
        'maximum_calibration_error': mce,
        'calibration_data': calibration_data
    }


def calculate_cv_metrics(cv_results: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    """
    Calculate cross-validation metrics statistics.
    
    Args:
        cv_results: List of metrics from each CV fold
        
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not cv_results:
        return {}
    
    # Get all metric names
    metric_names = set()
    for fold_results in cv_results:
        metric_names.update(fold_results.keys())
    
    cv_stats = {}
    
    for metric in metric_names:
        values = []
        for fold_results in cv_results:
            if metric in fold_results:
                values.append(fold_results[metric])
        
        if values:
            cv_stats[metric] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values),
                'values': values
            }
    
    return cv_stats


def calculate_bootstrap_metrics(y_true: np.ndarray, y_pred: np.ndarray, 
                              y_proba: Optional[np.ndarray] = None,
                              n_bootstrap: int = 1000, 
                              confidence_level: float = 0.95,
                              random_state: Optional[int] = None) -> Dict[str, Dict[str, float]]:
    """
    Calculate bootstrap confidence intervals for metrics.
    
    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities (optional)
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
        random_state: Random state for reproducibility
        
    Returns:
        Dictionary with bootstrap statistics for each metric
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    n_samples = len(y_true)
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    bootstrap_metrics = {}
    
    # Store bootstrap samples for each metric
    metric_samples = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f1': [],
        'roc_auc': [] if y_proba is not None else None
    }
    
    for i in range(n_bootstrap):
        # Bootstrap sample
        indices = np.random.choice(n_samples, size=n_samples, replace=True)
        y_true_boot = y_true[indices]
        y_pred_boot = y_pred[indices]
        y_proba_boot = y_proba[indices] if y_proba is not None else None
        
        # Calculate metrics for bootstrap sample
        try:
            boot_metrics = calculate_metrics(y_true_boot, y_pred_boot, y_proba_boot)
            
            metric_samples['accuracy'].append(boot_metrics['accuracy'])
            metric_samples['precision'].append(boot_metrics['precision'])
            metric_samples['recall'].append(boot_metrics['recall'])
            metric_samples['f1'].append(boot_metrics['f1'])
            
            if y_proba is not None and 'roc_auc' in boot_metrics:
                metric_samples['roc_auc'].append(boot_metrics['roc_auc'])
                
        except Exception:
            # Skip this bootstrap sample if metric calculation fails
            continue
    
    # Calculate confidence intervals
    for metric, samples in metric_samples.items():
        if samples and len(samples) > 0:
            bootstrap_metrics[metric] = {
                'mean': np.mean(samples),
                'std': np.std(samples),
                'lower_ci': np.percentile(samples, lower_percentile),
                'upper_ci': np.percentile(samples, upper_percentile),
                'median': np.median(samples)
            }
    
    return bootstrap_metrics


def calculate_threshold_metrics(y_true: np.ndarray, y_proba: np.ndarray, 
                              thresholds: Optional[np.ndarray] = None) -> pd.DataFrame:
    """
    Calculate metrics for different probability thresholds.
    
    Args:
        y_true: True labels
        y_proba: Predicted probabilities
        thresholds: Array of thresholds to evaluate (optional)
        
    Returns:
        DataFrame with metrics for each threshold
    """
    if thresholds is None:
        thresholds = np.arange(0.0, 1.01, 0.01)
    
    threshold_results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        try:
            metrics = calculate_metrics(y_true, y_pred_thresh, y_proba)
            metrics['threshold'] = threshold
            threshold_results.append(metrics)
        except Exception:
            # Skip this threshold if metric calculation fails
            continue
    
    return pd.DataFrame(threshold_results)


def compare_models(model_results: Dict[str, Dict[str, Any]], 
                  metrics_to_compare: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Compare multiple models' performance.
    
    Args:
        model_results: Dictionary with model names as keys and results as values
        metrics_to_compare: List of metrics to include in comparison
        
    Returns:
        DataFrame comparing models
    """
    if metrics_to_compare is None:
        metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
    
    comparison_data = []
    
    for model_name, results in model_results.items():
        row = {'model': model_name}
        
        for metric in metrics_to_compare:
            if metric in results:
                row[metric] = results[metric]
            else:
                row[metric] = np.nan
        
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Add ranking for each metric
    for metric in metrics_to_compare:
        if metric in comparison_df.columns:
            # Assume higher is better for all metrics except loss-based ones
            ascending = 'loss' in metric.lower()
            comparison_df[f'{metric}_rank'] = comparison_df[metric].rank(ascending=ascending)
    
    return comparison_df


def statistical_significance_test(y_true: np.ndarray, 
                                y_pred1: np.ndarray, 
                                y_pred2: np.ndarray,
                                test_type: str = 'mcnemar') -> Dict[str, float]:
    """
    Test statistical significance between two models.
    
    Args:
        y_true: True labels
        y_pred1: Predictions from model 1
        y_pred2: Predictions from model 2
        test_type: Type of test ('mcnemar' or 'paired_ttest')
        
    Returns:
        Dictionary with test results
    """
    if test_type == 'mcnemar':
        # McNemar's test for comparing two classifiers
        from scipy.stats import chi2
        
        # Create contingency table
        correct1 = (y_pred1 == y_true)
        correct2 = (y_pred2 == y_true)
        
        n01 = np.sum(~correct1 & correct2)  # Model 1 wrong, Model 2 correct
        n10 = np.sum(correct1 & ~correct2)  # Model 1 correct, Model 2 wrong
        
        # McNemar's test statistic
        if n01 + n10 == 0:
            p_value = 1.0
            statistic = 0.0
        else:
            statistic = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
            p_value = 1 - chi2.cdf(statistic, df=1)
        
        return {
            'test_type': 'mcnemar',
            'statistic': statistic,
            'p_value': p_value,
            'n01': n01,
            'n10': n10,
            'significant': p_value < 0.05
        }
    
    elif test_type == 'paired_ttest':
        # Paired t-test on accuracies (requires additional resampling)
        from scipy.stats import ttest_rel
        
        # This is a simplified version - ideally you'd need multiple samples
        acc1 = accuracy_score(y_true, y_pred1)
        acc2 = accuracy_score(y_true, y_pred2)
        
        # For demonstration - in practice you'd need bootstrap samples
        statistic, p_value = 0.0, 1.0  # Placeholder
        
        return {
            'test_type': 'paired_ttest',
            'statistic': statistic,
            'p_value': p_value,
            'acc1': acc1,
            'acc2': acc2,
            'significant': p_value < 0.05
        }
    
    else:
        raise ValueError(f"Unsupported test type: {test_type}")