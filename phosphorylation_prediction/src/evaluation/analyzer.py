"""Analysis tools for phosphorylation prediction results."""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import logging


class ErrorAnalyzer:
    """Analyze prediction errors and find patterns."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize error analyzer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def analyze_errors(self, y_true: np.ndarray, y_pred: np.ndarray, 
                      features: Optional[np.ndarray] = None,
                      additional_data: Optional[Dict[str, np.ndarray]] = None) -> Dict[str, Any]:
        """
        Analyze prediction errors comprehensively.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            features: Feature matrix (optional)
            additional_data: Additional data like sequences, positions (optional)
            
        Returns:
            Dictionary with error analysis results
        """
        analysis_results = {}
        
        # Basic error statistics
        errors = y_true != y_pred
        n_errors = np.sum(errors)
        error_rate = n_errors / len(y_true)
        
        analysis_results['error_statistics'] = {
            'total_errors': int(n_errors),
            'error_rate': float(error_rate),
            'total_samples': len(y_true)
        }
        
        # False positives and false negatives
        false_positives = (y_true == 0) & (y_pred == 1)
        false_negatives = (y_true == 1) & (y_pred == 0)
        
        analysis_results['error_types'] = {
            'false_positives': int(np.sum(false_positives)),
            'false_negatives': int(np.sum(false_negatives)),
            'fp_rate': float(np.sum(false_positives) / np.sum(y_true == 0)) if np.sum(y_true == 0) > 0 else 0.0,
            'fn_rate': float(np.sum(false_negatives) / np.sum(y_true == 1)) if np.sum(y_true == 1) > 0 else 0.0
        }
        
        # Feature-based error analysis
        if features is not None:
            feature_analysis = self._analyze_feature_errors(features, errors, y_true, y_pred)
            analysis_results['feature_analysis'] = feature_analysis
        
        # Sequence-based error analysis
        if additional_data is not None:
            sequence_analysis = self._analyze_sequence_errors(additional_data, errors, y_true, y_pred)
            analysis_results['sequence_analysis'] = sequence_analysis
        
        # Error clustering
        if features is not None and n_errors > 10:  # Only if we have enough errors
            clustering_results = self.cluster_errors(features[errors], n_clusters=min(5, n_errors//2))
            analysis_results['error_clustering'] = clustering_results
        
        return analysis_results
    
    def _analyze_feature_errors(self, features: np.ndarray, errors: np.ndarray,
                               y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze errors based on feature patterns."""
        feature_analysis = {}
        
        # Compare feature distributions for correct vs incorrect predictions
        correct_features = features[~errors]
        error_features = features[errors]
        
        if len(error_features) > 0 and len(correct_features) > 0:
            # Calculate feature statistics
            feature_analysis['feature_stats'] = {
                'correct_mean': np.mean(correct_features, axis=0).tolist(),
                'error_mean': np.mean(error_features, axis=0).tolist(),
                'correct_std': np.std(correct_features, axis=0).tolist(),
                'error_std': np.std(error_features, axis=0).tolist()
            }
            
            # Find features with largest differences
            mean_diff = np.abs(np.mean(error_features, axis=0) - np.mean(correct_features, axis=0))
            top_diff_indices = np.argsort(mean_diff)[-10:]  # Top 10 different features
            
            feature_analysis['discriminative_features'] = {
                'indices': top_diff_indices.tolist(),
                'differences': mean_diff[top_diff_indices].tolist()
            }
        
        return feature_analysis
    
    def _analyze_sequence_errors(self, additional_data: Dict[str, np.ndarray], 
                                errors: np.ndarray, y_true: np.ndarray, 
                                y_pred: np.ndarray) -> Dict[str, Any]:
        """Analyze errors based on sequence patterns."""
        sequence_analysis = {}
        
        # Analyze amino acid patterns in errors
        if 'sequences' in additional_data:
            sequences = additional_data['sequences']
            positions = additional_data.get('positions', None)
            
            error_sequences = sequences[errors]
            correct_sequences = sequences[~errors]
            
            # Extract central amino acids if positions are available
            if positions is not None:
                error_central_aa = []
                correct_central_aa = []
                
                for i, seq in enumerate(error_sequences):
                    if i < len(positions):
                        pos = positions[errors][i]
                        if 0 <= pos - 1 < len(seq):
                            error_central_aa.append(seq[pos - 1])
                
                for i, seq in enumerate(correct_sequences):
                    if i < len(positions):
                        pos = positions[~errors][i]
                        if 0 <= pos - 1 < len(seq):
                            correct_central_aa.append(seq[pos - 1])
                
                # Count amino acid frequencies
                if error_central_aa and correct_central_aa:
                    error_aa_counts = pd.Series(error_central_aa).value_counts()
                    correct_aa_counts = pd.Series(correct_central_aa).value_counts()
                    
                    sequence_analysis['amino_acid_patterns'] = {
                        'error_aa_distribution': error_aa_counts.to_dict(),
                        'correct_aa_distribution': correct_aa_counts.to_dict()
                    }
        
        # Analyze position-based patterns
        if 'positions' in additional_data:
            positions = additional_data['positions']
            error_positions = positions[errors]
            correct_positions = positions[~errors]
            
            sequence_analysis['position_patterns'] = {
                'error_position_stats': {
                    'mean': float(np.mean(error_positions)),
                    'std': float(np.std(error_positions)),
                    'min': int(np.min(error_positions)),
                    'max': int(np.max(error_positions))
                },
                'correct_position_stats': {
                    'mean': float(np.mean(correct_positions)),
                    'std': float(np.std(correct_positions)),
                    'min': int(np.min(correct_positions)),
                    'max': int(np.max(correct_positions))
                }
            }
        
        return sequence_analysis
    
    def find_error_patterns(self, errors: np.ndarray, features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Find patterns in prediction errors.
        
        Args:
            errors: Boolean array of errors
            features: Feature matrix
            
        Returns:
            List of detected patterns
        """
        patterns = []
        
        if np.sum(errors) == 0:
            return patterns
        
        error_features = features[errors]
        
        # Pattern 1: Outlier detection
        # Find samples with unusual feature values
        feature_means = np.mean(features, axis=0)
        feature_stds = np.std(features, axis=0)
        
        # Calculate z-scores for error samples
        z_scores = np.abs((error_features - feature_means) / (feature_stds + 1e-8))
        
        # Find samples with high z-scores
        outlier_threshold = 2.0
        outlier_samples = np.any(z_scores > outlier_threshold, axis=1)
        
        if np.sum(outlier_samples) > 0:
            patterns.append({
                'type': 'outliers',
                'description': f'Found {np.sum(outlier_samples)} error samples with outlier features',
                'sample_indices': np.where(errors)[0][outlier_samples].tolist(),
                'outlier_features': np.where(np.any(z_scores[outlier_samples] > outlier_threshold, axis=0))[0].tolist()
            })
        
        # Pattern 2: Feature range-based patterns
        # Find if errors are concentrated in specific feature ranges
        for feature_idx in range(features.shape[1]):
            feature_values = features[:, feature_idx]
            error_feature_values = error_features[:, feature_idx]
            
            # Check if errors are concentrated in high/low values
            feature_percentiles = np.percentile(feature_values, [25, 75])
            
            # Errors in extreme ranges
            low_range_errors = np.sum(error_feature_values < feature_percentiles[0])
            high_range_errors = np.sum(error_feature_values > feature_percentiles[1])
            
            total_errors = len(error_feature_values)
            
            if low_range_errors / total_errors > 0.6:  # 60% of errors in low range
                patterns.append({
                    'type': 'low_feature_range',
                    'description': f'Feature {feature_idx}: {low_range_errors}/{total_errors} errors in low range',
                    'feature_index': feature_idx,
                    'threshold': feature_percentiles[0]
                })
            
            if high_range_errors / total_errors > 0.6:  # 60% of errors in high range
                patterns.append({
                    'type': 'high_feature_range',
                    'description': f'Feature {feature_idx}: {high_range_errors}/{total_errors} errors in high range',
                    'feature_index': feature_idx,
                    'threshold': feature_percentiles[1]
                })
        
        return patterns
    
    def cluster_errors(self, error_features: np.ndarray, n_clusters: int = 5) -> Dict[str, Any]:
        """
        Cluster error samples to find similar error types.
        
        Args:
            error_features: Features of error samples
            n_clusters: Number of clusters
            
        Returns:
            Clustering results
        """
        if len(error_features) < n_clusters:
            n_clusters = max(1, len(error_features))
        
        # Perform clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(error_features)
        
        # Analyze clusters
        cluster_analysis = {
            'n_clusters': n_clusters,
            'cluster_labels': cluster_labels.tolist(),
            'cluster_centers': kmeans.cluster_centers_.tolist(),
            'cluster_sizes': []
        }
        
        for cluster_id in range(n_clusters):
            cluster_mask = cluster_labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            cluster_features = error_features[cluster_mask]
            
            cluster_info = {
                'cluster_id': cluster_id,
                'size': int(cluster_size),
                'percentage': float(cluster_size / len(error_features) * 100),
                'feature_mean': np.mean(cluster_features, axis=0).tolist(),
                'feature_std': np.std(cluster_features, axis=0).tolist()
            }
            
            cluster_analysis['cluster_sizes'].append(cluster_info)
        
        return cluster_analysis


class InterpretabilityAnalyzer:
    """Analyze model interpretability and feature importance."""
    
    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize interpretability analyzer.
        
        Args:
            logger: Optional logger instance
        """
        self.logger = logger or logging.getLogger(__name__)
    
    def calculate_shap_values(self, model, X: np.ndarray, 
                            sample_size: int = 100) -> np.ndarray:
        """
        Calculate SHAP values for model interpretability.
        
        Args:
            model: Trained model
            X: Input features
            sample_size: Number of samples to analyze
            
        Returns:
            SHAP values array
        """
        try:
            import shap
            
            # Sample data if too large
            if len(X) > sample_size:
                indices = np.random.choice(len(X), sample_size, replace=False)
                X_sample = X[indices]
            else:
                X_sample = X
            
            # Create explainer based on model type
            if hasattr(model, 'predict_proba'):
                # For tree-based models or sklearn models
                if hasattr(model, 'model') and hasattr(model.model, 'get_booster'):
                    # XGBoost model
                    explainer = shap.TreeExplainer(model.model)
                else:
                    # Generic model
                    explainer = shap.KernelExplainer(model.predict_proba, X_sample[:10])
            else:
                explainer = shap.KernelExplainer(model.predict, X_sample[:10])
            
            # Calculate SHAP values
            shap_values = explainer.shap_values(X_sample)
            
            # Handle different SHAP value formats
            if isinstance(shap_values, list):
                # Binary classification might return list
                shap_values = shap_values[1] if len(shap_values) == 2 else shap_values[0]
            
            return shap_values
            
        except ImportError:
            self.logger.warning("SHAP not available, skipping SHAP analysis")
            return None
        except Exception as e:
            self.logger.warning(f"Error calculating SHAP values: {e}")
            return None
    
    def analyze_attention_patterns(self, transformer_model, sequences: List[str],
                                 positions: List[int]) -> Dict[str, Any]:
        """
        Analyze attention patterns in transformer models.
        
        Args:
            transformer_model: Trained transformer model
            sequences: List of protein sequences
            positions: List of positions in sequences
            
        Returns:
            Attention analysis results
        """
        attention_analysis = {}
        
        try:
            if not hasattr(transformer_model, 'get_attention_weights'):
                self.logger.warning("Model does not support attention analysis")
                return attention_analysis
            
            attention_patterns = []
            
            # Analyze attention for sample sequences
            sample_size = min(50, len(sequences))
            for i in range(sample_size):
                seq = sequences[i]
                pos = positions[i]
                
                attention_weights = transformer_model.get_attention_weights(seq, pos)
                
                if attention_weights is not None:
                    attention_patterns.append({
                        'sequence_idx': i,
                        'sequence_length': len(seq),
                        'target_position': pos,
                        'attention_weights': attention_weights.tolist(),
                        'max_attention_pos': int(np.argmax(np.mean(attention_weights, axis=0))),
                        'attention_entropy': float(-np.sum(attention_weights * np.log(attention_weights + 1e-8)))
                    })
            
            attention_analysis['patterns'] = attention_patterns
            
            # Aggregate statistics
            if attention_patterns:
                entropies = [p['attention_entropy'] for p in attention_patterns]
                attention_analysis['statistics'] = {
                    'mean_entropy': float(np.mean(entropies)),
                    'std_entropy': float(np.std(entropies)),
                    'n_analyzed': len(attention_patterns)
                }
            
        except Exception as e:
            self.logger.warning(f"Error analyzing attention patterns: {e}")
        
        return attention_analysis
    
    def compare_model_decisions(self, models: List, X: np.ndarray) -> pd.DataFrame:
        """
        Compare decision boundaries and agreements between models.
        
        Args:
            models: List of trained models
            X: Input features
            
        Returns:
            DataFrame with model comparison results
        """
        if len(models) < 2:
            raise ValueError("Need at least 2 models for comparison")
        
        # Get predictions from all models
        predictions = {}
        probabilities = {}
        
        for i, model in enumerate(models):
            model_name = f"model_{i}"
            
            if hasattr(model, 'predict'):
                predictions[model_name] = model.predict(X)
            
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X)
                if probs.ndim == 2:
                    probs = probs[:, 1]  # Take positive class probabilities
                probabilities[model_name] = probs
        
        # Create comparison DataFrame
        comparison_data = {'sample_idx': range(len(X))}
        
        # Add predictions
        for model_name, preds in predictions.items():
            comparison_data[f'{model_name}_pred'] = preds
        
        # Add probabilities
        for model_name, probs in probabilities.items():
            comparison_data[f'{model_name}_prob'] = probs
        
        df = pd.DataFrame(comparison_data)
        
        # Calculate agreement statistics
        if len(predictions) >= 2:
            model_names = list(predictions.keys())
            
            # Pairwise agreements
            for i in range(len(model_names)):
                for j in range(i + 1, len(model_names)):
                    model1, model2 = model_names[i], model_names[j]
                    agreement = predictions[model1] == predictions[model2]
                    df[f'{model1}_{model2}_agree'] = agreement
            
            # Overall agreement (all models agree)
            pred_matrix = np.column_stack([predictions[name] for name in model_names])
            all_agree = np.all(pred_matrix == pred_matrix[:, 0:1], axis=1)
            df['all_models_agree'] = all_agree
        
        return df
    
    def analyze_feature_interactions(self, features: np.ndarray, 
                                   feature_names: Optional[List[str]] = None,
                                   top_k: int = 20) -> Dict[str, Any]:
        """
        Analyze feature interactions and correlations.
        
        Args:
            features: Feature matrix
            feature_names: List of feature names (optional)
            top_k: Number of top interactions to return
            
        Returns:
            Feature interaction analysis
        """
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(features.shape[1])]
        
        # Calculate correlation matrix
        correlation_matrix = np.corrcoef(features.T)
        
        # Find strong correlations (excluding diagonal)
        correlation_pairs = []
        n_features = len(feature_names)
        
        for i in range(n_features):
            for j in range(i + 1, n_features):
                corr_value = correlation_matrix[i, j]
                correlation_pairs.append({
                    'feature1': feature_names[i],
                    'feature2': feature_names[j],
                    'correlation': float(corr_value),
                    'abs_correlation': float(abs(corr_value))
                })
        
        # Sort by absolute correlation
        correlation_pairs.sort(key=lambda x: x['abs_correlation'], reverse=True)
        
        # Principal Component Analysis for feature importance
        try:
            pca = PCA()
            pca.fit(features)
            
            # Get feature loadings for first few components
            n_components = min(5, features.shape[1])
            loadings = pca.components_[:n_components].T
            
            # Calculate feature importance as sum of absolute loadings weighted by explained variance
            explained_variance = pca.explained_variance_ratio_[:n_components]
            feature_importance = np.sum(np.abs(loadings) * explained_variance, axis=1)
            
            pca_analysis = {
                'explained_variance_ratio': explained_variance.tolist(),
                'feature_importance': dict(zip(feature_names, feature_importance.tolist())),
                'n_components_90pct': int(np.argmax(np.cumsum(pca.explained_variance_ratio_) >= 0.9)) + 1
            }
            
        except Exception as e:
            self.logger.warning(f"PCA analysis failed: {e}")
            pca_analysis = None
        
        return {
            'correlation_analysis': {
                'top_correlations': correlation_pairs[:top_k],
                'correlation_matrix': correlation_matrix.tolist()
            },
            'pca_analysis': pca_analysis
        }
    
    def generate_interpretability_report(self, model, X: np.ndarray, y: np.ndarray,
                                       feature_names: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Generate comprehensive interpretability report.
        
        Args:
            model: Trained model
            X: Features
            y: Labels
            feature_names: Feature names (optional)
            
        Returns:
            Comprehensive interpretability report
        """
        report = {}
        
        # Model-specific feature importance
        if hasattr(model, 'get_feature_importance'):
            try:
                importance = model.get_feature_importance()
                report['model_feature_importance'] = importance
            except Exception as e:
                self.logger.warning(f"Could not get model feature importance: {e}")
        
        # SHAP analysis
        shap_values = self.calculate_shap_values(model, X, sample_size=100)
        if shap_values is not None:
            # Calculate mean absolute SHAP values
            mean_shap = np.mean(np.abs(shap_values), axis=0)
            
            if feature_names is not None:
                shap_importance = dict(zip(feature_names, mean_shap))
            else:
                shap_importance = {f"feature_{i}": val for i, val in enumerate(mean_shap)}
            
            report['shap_analysis'] = {
                'feature_importance': shap_importance,
                'shap_values_sample': shap_values[:10].tolist()  # First 10 samples
            }
        
        # Feature interaction analysis
        if feature_names is not None:
            interaction_analysis = self.analyze_feature_interactions(X, feature_names)
            report['feature_interactions'] = interaction_analysis
        
        # Prediction confidence analysis
        if hasattr(model, 'predict_proba'):
            y_proba = model.predict_proba(X)
            if y_proba.ndim == 2:
                y_proba = y_proba[:, 1]
            
            # Analyze confidence distribution
            high_conf_correct = np.sum((y_proba > 0.8) & (model.predict(X) == y))
            high_conf_total = np.sum(y_proba > 0.8)
            low_conf_correct = np.sum((y_proba < 0.2) & (model.predict(X) == y))
            low_conf_total = np.sum(y_proba < 0.2)
            
            report['confidence_analysis'] = {
                'high_confidence_accuracy': float(high_conf_correct / high_conf_total) if high_conf_total > 0 else 0.0,
                'low_confidence_accuracy': float(low_conf_correct / low_conf_total) if low_conf_total > 0 else 0.0,
                'mean_confidence': float(np.mean(np.maximum(y_proba, 1 - y_proba))),
                'confidence_distribution': {
                    'high_conf_samples': int(high_conf_total),
                    'medium_conf_samples': int(np.sum((y_proba >= 0.2) & (y_proba <= 0.8))),
                    'low_conf_samples': int(low_conf_total)
                }
            }
        
        return report