"""
FedBN Aggregator for ECG Federated Learning.

FedBN (Federated Learning with Batch Normalization) is designed for models
with BatchNorm layers. It aggregates all parameters EXCEPT:
- running_mean
- running_var
- num_batches_tracked

These BatchNorm statistics remain local to each client.

Reference: Li et al., "FedBN: Federated Learning on Non-IID Features via Local Batch Normalization"
"""

import numpy as np
import torch
import os
from app.fl.ecg_model import ECGModel


class FedBNAggregator:
    """
    FedBN Aggregator for ECG model with ResNet1D-18.
    
    Key difference from FedAvg: BatchNorm statistics (running_mean, running_var)
    are NOT aggregated - they remain client-specific.
    """
    
    MAX_WEIGHT_NORM = 50.0  # Maximum allowed L2 norm per parameter tensor
    
    def __init__(self):
        """Initialize FedBN aggregator for binary ECG classification."""
        # Binary classification: Normal vs Arrhythmia
        self.global_model = ECGModel(num_classes=2, in_channels=12, attention=True)
        self.round = 0
        self.client_weights = []
        self.client_sizes = []
        self.client_metrics = []
        self.history = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'num_clients': []
        }
        
        # Track that this is FedBN aggregation
        self.aggregation_type = 'fedbn'
    
    def reset(self):
        """Reset the aggregator state completely."""
        # Reset with binary classification model
        self.global_model = ECGModel(num_classes=2, in_channels=12, attention=True)
        self.round = 0
        self.client_weights = []
        self.client_sizes = []
        self.client_metrics = []
        self.history = {
            'rounds': [],
            'accuracy': [],
            'loss': [],
            'num_clients': []
        }
    
    def initialize_global_model(self):
        """Initialize global model (random weights from ECGModel init)."""
        return True
    
    def _is_batchnorm_stat(self, key: str) -> bool:
        """
        Check if parameter is BatchNorm statistic (not aggregated).
        
        Args:
            key: Parameter name from state_dict
            
        Returns:
            True if parameter is running_mean, running_var, or num_batches_tracked
        """
        return any(stat in key for stat in [
            'running_mean',
            'running_var',
            'num_batches_tracked'
        ])
    
    def _validate_weights(self, weights: dict):
        """
        Validate client weights against global model structure.
        
        Args:
            weights: Client state dict
        """
        global_weights = self.global_model.get_weights()
        
        # Check that all keys match
        if set(weights.keys()) != set(global_weights.keys()):
            raise ValueError(
                f"Weight keys mismatch. Expected {set(global_weights.keys())}, "
                f"got {set(weights.keys())}"
            )
        
        # Check shapes and types (skip NaN/Inf check for BatchNorm stats)
        for key in global_weights:
            if weights[key].shape != global_weights[key].shape:
                raise ValueError(
                    f"Shape mismatch for '{key}': expected {global_weights[key].shape}, "
                    f"got {weights[key].shape}"
                )
            if not torch.is_floating_point(weights[key]) and 'num_batches_tracked' not in key:
                raise ValueError(f"Non-floating-point tensor for '{key}'")
            # Only check NaN/Inf for parameters that will be aggregated
            if not self._is_batchnorm_stat(key):
                if torch.isnan(weights[key]).any() or torch.isinf(weights[key]).any():
                    raise ValueError(f"NaN or Inf detected in weights for '{key}'")
    
    def _clip_weights(self, weights: dict) -> dict:
        """
        Clip weight tensors to a maximum L2 norm to prevent poisoning.
        
        Args:
            weights: Client weights
            
        Returns:
            Clipped weights
        """
        clipped = {}
        for key, tensor in weights.items():
            # Skip clipping for BatchNorm stats (not aggregated anyway)
            if self._is_batchnorm_stat(key):
                clipped[key] = tensor
                continue
            
            norm = torch.norm(tensor.float()).item()
            if norm > self.MAX_WEIGHT_NORM:
                clipped[key] = tensor * (self.MAX_WEIGHT_NORM / norm)
            else:
                clipped[key] = tensor
        return clipped
    
    def aggregate(self) -> bool:
        """
        FedBN Aggregation.
        
        For each parameter:
        - If BatchNorm statistic (running_mean, running_var): Skip aggregation
          (keep global model's current statistics)
        - Otherwise: Apply FedAvg weighted by number of samples
        
        Formula:
        w_global = sum(n_k * w_k) / sum(n_k)  [for non-BN params]
        
        Returns:
            True if aggregation successful
        """
        if not self.client_weights:
            return False
        
        total_samples = sum(self.client_sizes)
        
        # Initialize aggregates
        first_weights = self.client_weights[0]
        agg_weights = {}
        
        # Process each parameter
        for key in first_weights.keys():
            # Skip BatchNorm statistics in aggregation
            if self._is_batchnorm_stat(key):
                # Keep current global model's BatchNorm statistics
                # (they remain unchanged during aggregation)
                continue
            
            # Initialize aggregation for this parameter
            agg_weights[key] = torch.zeros_like(first_weights[key])
            
            # Weighted aggregation (FedAvg)
            for weights, n_samples in zip(self.client_weights, self.client_sizes):
                agg_weights[key] += weights[key] * n_samples
            
            # Average by total samples
            agg_weights[key] /= total_samples
        
        # Update global model with aggregated weights
        # Get current global weights
        current_weights = self.global_model.get_weights()
        
        # Merge: aggregated params + keep BN stats from global model
        final_weights = {}
        for key in current_weights.keys():
            if self._is_batchnorm_stat(key):
                # Keep global model's BatchNorm statistics
                final_weights[key] = current_weights[key]
            else:
                # Use aggregated weights
                final_weights[key] = agg_weights[key]
        
        self.global_model.set_weights(final_weights)
        
        # Compute weighted average of client metrics
        if self.client_metrics:
            weighted_acc = sum(
                m.get('accuracy', 0) * n for m, n in zip(self.client_metrics, self.client_sizes)
            ) / total_samples
            weighted_loss = sum(
                m.get('loss', 0) * n for m, n in zip(self.client_metrics, self.client_sizes)
            ) / total_samples
        else:
            weighted_acc = 0.0
            weighted_loss = 0.0
        
        # Update history
        self.history['rounds'].append(self.round + 1)
        self.history['accuracy'].append(round(weighted_acc, 4))
        self.history['loss'].append(round(weighted_loss, 4))
        self.history['num_clients'].append(len(self.client_weights))
        
        # Audit Log
        try:
            from app import db
            from app.models import AuditLog
            n_clients = len(self.client_weights)
            log = AuditLog(
                action='ECG FL Round Aggregation (FedBN)',
                details=f'Round {self.round+1} completed. '
                        f'{n_clients} client(s), {total_samples} total samples. '
                        f'Avg Accuracy: {weighted_acc:.4f}, Avg Loss: {weighted_loss:.4f}'
            )
            db.session.add(log)
            db.session.commit()
        except:
            pass  # Avoid breaking FL if DB fails
        
        # Save Global Model
        try:
            save_dir = os.path.join(os.path.dirname(__file__), 'saved_models')
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            
            # Save versioned round
            versioned_path = os.path.join(save_dir, f'ecg_model_round_{self.round}.pkl')
            self.global_model.save(versioned_path)
            
            # Save latest
            latest_path = os.path.join(save_dir, 'ecg_model_latest.pkl')
            self.global_model.save(latest_path)
            
        except Exception as e:
            print(f"Error saving ECG model: {e}")
        
        # Clear for next round
        self.client_weights = []
        self.client_sizes = []
        self.client_metrics = []
        self.round += 1
        
        return True
    
    def add_client_update(
        self,
        weights: dict,
        n_samples: int,
        metrics: dict = None
    ):
        """
        Store a client's trained weights and metrics.
        
        Args:
            weights: Client state dict (includes all params, BN stats ignored in aggregation)
            n_samples: Number of samples client trained on
            metrics: Optional dict with 'accuracy', 'loss'
        """
        # Validate weight structure
        self._validate_weights(weights)
        
        # Clip weights to prevent poisoning (only for non-BN params)
        clipped_weights = self._clip_weights(weights)
        
        self.client_weights.append(clipped_weights)
        self.client_sizes.append(n_samples)
        self.client_metrics.append(metrics or {})
    
    def get_global_model(self) -> ECGModel:
        """Get current global model."""
        return self.global_model
    
    def get_status(self) -> dict:
        """Get current aggregator status."""
        return {
            'round': self.round,
            'aggregation_type': self.aggregation_type,
            'clients_updated': len(self.client_weights),
            'clients_waiting': len(self.client_weights) > 0,
            'total_samples': sum(self.client_sizes)
        }
