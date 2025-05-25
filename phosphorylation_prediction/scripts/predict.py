#!/usr/bin/env python3
"""Prediction script for phosphorylation prediction models."""

import argparse
import sys
import pandas as pd
import numpy as np
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from config import load_config
from data import SequenceDataset, SequenceProcessor, FeatureExtractor
from utils import get_logger, load_experiment_checkpoint, ModelSaver
import torch

def main():
    parser = argparse.ArgumentParser(description="Make predictions with trained models")
    parser.add_argument("--model-path", type=str, required=True,
                       help="Path to trained model or experiment checkpoint")
    parser.add_argument("--input-file", type=str, required=True,
                       help="Path to input sequences (CSV or FASTA)")
    parser.add_argument("--output-file", type=str, required=True,
                       help="Path to output predictions")
    parser.add_argument("--config", type=str,
                       help="Path to configuration file")
    parser.add_argument("--model-type", type=str,
                       choices=["xgboost", "transformer", "ensemble"],
                       help="Model type (if not in checkpoint)")
    parser.add_argument("--batch-size", type=int, default=32,
                       help="Batch size for prediction")
    parser.add_argument("--device", type=str, default="auto",
                       choices=["cpu", "cuda", "auto"],
                       help="Device to use for prediction")
    parser.add_argument("--include-features", action="store_true",
                       help="Include extracted features in output")
    parser.add_argument("--confidence-threshold", type=float, default=0.5,
                       help="Confidence threshold for binary predictions")
    
    args = parser.parse_args()
    
    # Setup
    logger = get_logger(__name__)
    
    # Determine device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    logger.info(f"Using device: {device}")
    
    try:
        # Load model and config
        model_path = Path(args.model_path)
        
        if model_path.suffix == ".json":
            # Experiment checkpoint
            logger.info("Loading experiment checkpoint...")
            checkpoint_data = load_experiment_checkpoint(model_path)
            config = checkpoint_data['config']
            models = checkpoint_data['models']
            
            if len(models) == 1:
                model = list(models.values())[0]
                model_type = list(models.keys())[0]
            else:
                # Ensemble
                model = models
                model_type = "ensemble"
        
        else:
            # Individual model
            if not args.config:
                raise ValueError("--config required when loading individual model")
            
            config = load_config(args.config)
            model_type = args.model_type
            
            if model_type == "xgboost":
                model = ModelSaver.load_sklearn_model(model_path)
            elif model_type == "transformer":
                # Need to reconstruct model architecture
                raise NotImplementedError("Direct transformer loading not implemented. Use experiment checkpoint.")
            elif model_type == "ensemble":
                model = ModelSaver.load_ensemble_model(model_path)
        
        logger.info(f"Loaded {model_type} model from {model_path}")
        
        # Load and process input data
        logger.info(f"Loading input data from {args.input_file}")
        
        input_path = Path(args.input_file)
        if input_path.suffix.lower() == ".csv":
            # Assume CSV with 'sequence' column
            df = pd.read_csv(input_path)
            sequences = df['sequence'].tolist()
            sequence_ids = df.get('id', range(len(sequences))).tolist()
        else:
            # Assume FASTA format
            sequences = []
            sequence_ids = []
            current_id = None
            current_seq = ""
            
            with open(input_path, 'r') as f:
                for line in f:
                    line = line.strip()
                    if line.startswith('>'):
                        if current_id is not None:
                            sequences.append(current_seq)
                            sequence_ids.append(current_id)
                        current_id = line[1:]
                        current_seq = ""
                    else:
                        current_seq += line
                
                if current_id is not None:
                    sequences.append(current_seq)
                    sequence_ids.append(current_id)
        
        logger.info(f"Loaded {len(sequences)} sequences")
        
        # Create dataset
        dataset = SequenceDataset(sequences, labels=None)
        
        # Process sequences
        processor = SequenceProcessor(
            window_size=config['data']['window_size'],
            tokenizer_name=config['data'].get('tokenizer_name')
        )
        
        processed_sequences = []
        all_windows = []
        window_positions = []
        
        for i, sequence in enumerate(sequences):
            windows, positions = processor.extract_windows(sequence)
            processed_sequences.extend([sequence] * len(windows))
            all_windows.extend(windows)
            window_positions.extend([(i, pos) for pos in positions])
        
        logger.info(f"Extracted {len(all_windows)} windows for prediction")
        
        # Make predictions
        logger.info("Making predictions...")
        
        if model_type in ["xgboost", "ensemble"]:
            # Extract features
            feature_extractor = FeatureExtractor()
            features = feature_extractor.extract_features(all_windows)
            
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(features)
                if probabilities.ndim == 2 and probabilities.shape[1] == 2:
                    probabilities = probabilities[:, 1]  # Positive class
            else:
                probabilities = model.predict(features)
        
        elif model_type == "transformer":
            # Tokenize sequences
            tokenized = processor.tokenize_sequences(all_windows)
            
            # Create DataLoader
            from torch.utils.data import DataLoader, TensorDataset
            dataset = TensorDataset(
                torch.tensor(tokenized['input_ids']),
                torch.tensor(tokenized['attention_mask'])
            )
            dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
            
            # Predict
            model.eval()
            model.to(device)
            probabilities = []
            
            with torch.no_grad():
                for batch in dataloader:
                    input_ids, attention_mask = [b.to(device) for b in batch]
                    outputs = model(input_ids, attention_mask)
                    probs = torch.softmax(outputs.logits, dim=-1)[:, 1]  # Positive class
                    probabilities.extend(probs.cpu().numpy())
            
            probabilities = np.array(probabilities)
        
        # Create binary predictions
        binary_predictions = (probabilities >= args.confidence_threshold).astype(int)
        
        # Organize results by sequence
        results = []
        for (seq_idx, position), prob, binary_pred in zip(window_positions, probabilities, binary_predictions):
            result = {
                'sequence_id': sequence_ids[seq_idx],
                'sequence': sequences[seq_idx],
                'position': position,
                'window': all_windows[window_positions.index((seq_idx, position))],
                'probability': float(prob),
                'prediction': int(binary_pred),
                'confident': 'Yes' if abs(prob - 0.5) > 0.3 else 'No'
            }
            
            if args.include_features and model_type in ["xgboost", "ensemble"]:
                feature_names = feature_extractor.get_feature_names()
                feature_idx = window_positions.index((seq_idx, position))
                for feat_name, feat_value in zip(feature_names, features[feature_idx]):
                    result[f'feature_{feat_name}'] = float(feat_value)
            
            results.append(result)
        
        # Save results
        results_df = pd.DataFrame(results)
        
        # Sort by sequence and position
        results_df = results_df.sort_values(['sequence_id', 'position'])
        
        output_path = Path(args.output_file)
        results_df.to_csv(output_path, index=False)
        
        logger.info(f"Predictions saved to {output_path}")
        
        # Print summary
        total_sites = len(results_df)
        positive_sites = (results_df['prediction'] == 1).sum()
        confident_sites = (results_df['confident'] == 'Yes').sum()
        
        logger.info(f"Summary:")
        logger.info(f"  Total sites: {total_sites}")
        logger.info(f"  Predicted positive: {positive_sites} ({positive_sites/total_sites*100:.1f}%)")
        logger.info(f"  Confident predictions: {confident_sites} ({confident_sites/total_sites*100:.1f}%)")
        logger.info(f"  Average probability: {results_df['probability'].mean():.3f}")
    
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        sys.exit(1)

if __name__ == "__main__":
    main()