"""Transformer model implementation for phosphorylation prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModel, get_linear_schedule_with_warmup
from torch.optim import AdamW
from torch.utils.data import DataLoader
from typing import Dict, Any, Optional, Union, Tuple
import logging
import os
import json
from .base_model import BaseModel


class PhosTransformer(nn.Module):
    """Transformer architecture for phosphorylation prediction."""
    
    def __init__(self, config: Dict[str, Any]):
        """
        Initialize PhosTransformer.
        
        Args:
            config: Model configuration
        """
        super().__init__()
        
        self.config = config
        model_name = config.get('model_name', 'facebook/esm2_t6_8M_UR50D')
        dropout_rate = config.get('dropout_rate', 0.3)
        window_context = config.get('window_context', 3)
        
        # Load pre-trained protein language model
        self.protein_encoder = AutoModel.from_pretrained(model_name)
        
        # Get hidden size from the model config
        hidden_size = self.protein_encoder.config.hidden_size
        
        # Context aggregation (lightweight)
        self.window_context = window_context
        context_size = hidden_size * (2 * window_context + 1)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(context_size, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 64),
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, 1)
        )
        
    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the model.
        
        Args:
            input_ids: Token IDs
            attention_mask: Attention mask
            
        Returns:
            Logits for binary classification
        """
        # Get the transformer outputs
        outputs = self.protein_encoder(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        # Get sequence outputs
        sequence_output = outputs.last_hidden_state  # [batch_size, seq_len, hidden_size]
        
        # Find the center position
        center_pos = sequence_output.shape[1] // 2
        
        # Extract features from window around center
        batch_size, seq_len, hidden_dim = sequence_output.shape
        context_features = []
        
        for i in range(-self.window_context, self.window_context + 1):
            pos = center_pos + i
            # Handle boundary cases
            if pos < 0 or pos >= seq_len:
                # Use zero padding for out-of-bounds positions
                context_features.append(torch.zeros(batch_size, hidden_dim, device=sequence_output.device))
            else:
                context_features.append(sequence_output[:, pos, :])
        
        # Concatenate context features
        concat_features = torch.cat(context_features, dim=1)
        
        # Pass through classifier
        logits = self.classifier(concat_features)
        
        return logits.squeeze(-1)


class TransformerModel(BaseModel):
    """Transformer wrapper implementing BaseModel interface."""
    
    def __init__(self, config: Dict[str, Any], logger: Optional[logging.Logger] = None):
        """
        Initialize Transformer model.
        
        Args:
            config: Model configuration
            logger: Optional logger instance
        """
        super().__init__(config, logger)
        
        self.model_name = config.get('model_name', 'facebook/esm2_t6_8M_UR50D')
        self.device = torch.device(config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu'))
        self.max_length = config.get('max_length', 64)
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = PhosTransformer(config).to(self.device)
        
        # Training parameters
        self.learning_rate = config.get('learning_rate', 2e-5)
        self.num_epochs = config.get('num_epochs', 10)
        self.batch_size = config.get('batch_size', 32)
        self.weight_decay = config.get('weight_decay', 0.01)
        self.warmup_steps = config.get('warmup_steps', 500)
        self.gradient_accumulation_steps = config.get('gradient_accumulation_steps', 4)
        self.fp16 = config.get('fp16', True)
        
        # Initialize optimizer and scheduler (will be set during training)
        self.optimizer = None
        self.scheduler = None
        
        # Training history
        self.training_history = {'train': [], 'val': []}
        
        self.logger.info(f"Initialized Transformer model: {self.model_name}")
        self.logger.info(f"Device: {self.device}")
    
    def fit(self, train_loader: DataLoader, val_loader: Optional[DataLoader] = None, 
            **kwargs) -> 'TransformerModel':
        """
        Train transformer with mixed precision.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            **kwargs: Additional training arguments
            
        Returns:
            Self for method chaining
        """
        self.logger.info("Starting Transformer training...")
        
        # Set up optimizer
        self.optimizer = AdamW(
            self.model.parameters(), 
            lr=self.learning_rate, 
            weight_decay=self.weight_decay
        )
        
        # Calculate total steps for scheduler
        total_steps = len(train_loader) * self.num_epochs
        
        # Set up scheduler
        self.scheduler = get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=self.warmup_steps,
            num_training_steps=total_steps
        )
        
        # Set up mixed precision training
        scaler = torch.cuda.amp.GradScaler() if self.fp16 and self.device.type == 'cuda' else None
        
        # Training loop
        for epoch in range(self.num_epochs):
            self.logger.info(f"Epoch {epoch+1}/{self.num_epochs}")
            
            # Train
            train_metrics = self._train_epoch(train_loader, scaler)
            self.training_history['train'].append(train_metrics)
            
            # Validate
            if val_loader is not None:
                val_metrics = self._evaluate_epoch(val_loader)
                self.training_history['val'].append(val_metrics)
                
                self.logger.info(
                    f"Train Loss: {train_metrics['loss']:.4f}, "
                    f"Val Loss: {val_metrics['loss']:.4f}, "
                    f"Val Acc: {val_metrics['accuracy']:.4f}, "
                    f"Val F1: {val_metrics['f1']:.4f}"
                )
            else:
                self.logger.info(f"Train Loss: {train_metrics['loss']:.4f}")
        
        self.is_fitted = True
        self.logger.info("Transformer training completed")
        
        return self
    
    def _train_epoch(self, data_loader: DataLoader, scaler) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_targets = []
        all_predictions = []
        
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            targets = batch['target'].to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass with mixed precision
            if scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(input_ids, attention_mask)
                    loss = F.binary_cross_entropy_with_logits(outputs, targets)
                
                # Backward pass with scaling
                scaler.scale(loss).backward()
                
                # Gradient accumulation
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    scaler.step(self.optimizer)
                    scaler.update()
                    self.scheduler.step()
            else:
                # Regular training without mixed precision
                outputs = self.model(input_ids, attention_mask)
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                
                loss.backward()
                
                if (batch_idx + 1) % self.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                    self.scheduler.step()
            
            # Accumulate loss
            total_loss += loss.item()
            
            # Store predictions and targets for metrics
            with torch.no_grad():
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        
        avg_loss = total_loss / len(data_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def _evaluate_epoch(self, data_loader: DataLoader) -> Dict[str, float]:
        """Evaluate for one epoch."""
        self.model.eval()
        total_loss = 0
        all_targets = []
        all_predictions = []
        all_probabilities = []
        
        with torch.no_grad():
            for batch in data_loader:
                # Get batch data
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                targets = batch['target'].to(self.device)
                
                # Forward pass
                outputs = self.model(input_ids, attention_mask)
                
                # Calculate loss
                loss = F.binary_cross_entropy_with_logits(outputs, targets)
                total_loss += loss.item()
                
                # Store predictions and targets for metrics
                probs = torch.sigmoid(outputs)
                preds = (probs > 0.5).float()
                
                all_targets.extend(targets.cpu().numpy())
                all_predictions.extend(preds.cpu().numpy())
                all_probabilities.extend(probs.cpu().numpy())
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
        
        accuracy = accuracy_score(all_targets, all_predictions)
        precision = precision_score(all_targets, all_predictions, zero_division=0)
        recall = recall_score(all_targets, all_predictions, zero_division=0)
        f1 = f1_score(all_targets, all_predictions, zero_division=0)
        auc = roc_auc_score(all_targets, all_probabilities)
        
        avg_loss = total_loss / len(data_loader)
        
        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
    
    def predict(self, X: Union[DataLoader, torch.Tensor]) -> np.ndarray:
        """
        Return class predictions.
        
        Args:
            X: Input data (DataLoader or tensor)
            
        Returns:
            Predicted classes
        """
        probabilities = self.predict_proba(X)
        return (probabilities > 0.5).astype(int)
    
    def predict_proba(self, X: Union[DataLoader, torch.Tensor]) -> np.ndarray:
        """
        Return probability predictions.
        
        Args:
            X: Input data (DataLoader or tensor)
            
        Returns:
            Predicted probabilities
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before making predictions")
        
        self.model.eval()
        all_probabilities = []
        
        with torch.no_grad():
            if isinstance(X, DataLoader):
                for batch in X:
                    input_ids = batch['input_ids'].to(self.device)
                    attention_mask = batch['attention_mask'].to(self.device)
                    
                    outputs = self.model(input_ids, attention_mask)
                    probs = torch.sigmoid(outputs)
                    all_probabilities.extend(probs.cpu().numpy())
            else:
                # Handle tensor input (for compatibility)
                if isinstance(X, torch.Tensor):
                    X = X.to(self.device)
                    outputs = self.model(X)
                    probs = torch.sigmoid(outputs)
                    all_probabilities = probs.cpu().numpy()
        
        return np.array(all_probabilities)
    
    def save(self, path: str) -> None:
        """
        Save model to disk.
        
        Args:
            path: Path to save model
        """
        if not self.is_fitted:
            raise ValueError("Cannot save unfitted model")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Save model state dict
        model_path = path if path.endswith('.pt') else f"{path}.pt"
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'training_history': self.training_history
        }, model_path)
        
        # Save tokenizer
        tokenizer_path = path.replace('.pt', '_tokenizer')
        self.tokenizer.save_pretrained(tokenizer_path)
        
        self.logger.info(f"Model saved to {model_path}")
        self.logger.info(f"Tokenizer saved to {tokenizer_path}")
    
    def load(self, path: str) -> None:
        """
        Load model from disk.
        
        Args:
            path: Path to load model from
        """
        model_path = path if path.endswith('.pt') else f"{path}.pt"
        
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")
        
        # Load checkpoint
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Update config and create model
        self.config.update(checkpoint.get('config', {}))
        self.model = PhosTransformer(self.config).to(self.device)
        
        # Load model state
        self.model.load_state_dict(checkpoint['model_state_dict'])
        
        # Load training history
        self.training_history = checkpoint.get('training_history', {'train': [], 'val': []})
        
        # Load tokenizer
        tokenizer_path = path.replace('.pt', '_tokenizer')
        if os.path.exists(tokenizer_path):
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
        
        self.is_fitted = True
        self.logger.info(f"Model loaded from {model_path}")
    
    def get_attention_weights(self, sequence: str, position: int) -> np.ndarray:
        """
        Extract attention weights for interpretability.
        
        Args:
            sequence: Input protein sequence
            position: Position in sequence
            
        Returns:
            Attention weights
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted to extract attention weights")
        
        # Extract window around position
        from ..data.preprocessor import SequenceProcessor
        processor = SequenceProcessor(self.config.get('window_size', 10))
        window = processor.extract_window(sequence, position)
        
        # Tokenize
        encoding = self.tokenizer(
            window,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)
        
        # Get attention weights from the model
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.protein_encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True
            )
            
            # Extract attention weights from last layer
            attentions = outputs.attentions[-1]  # Last layer
            
            # Average over heads and batch dimension
            attention_weights = attentions.mean(dim=1).squeeze(0)  # [seq_len, seq_len]
            
        return attention_weights.cpu().numpy()
    
    def get_training_history(self) -> Dict[str, Any]:
        """
        Get training history.
        
        Returns:
            Training history dictionary
        """
        return self.training_history
    
    def __repr__(self) -> str:
        """String representation of the Transformer model."""
        status = "fitted" if self.is_fitted else "unfitted"
        return f"TransformerModel(model={self.model_name}, status={status}, device={self.device})"