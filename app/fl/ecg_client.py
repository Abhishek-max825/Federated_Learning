"""
ECG Federated Learning Client

This client runs on hospital nodes for local ECG model training.

Flow:
    1. Receive global model weights from server
    2. Load local hospital ECG dataset (WFDB files)
    3. Preprocess signals (unified pipeline)
    4. Train local model
    5. Send weight updates (not data) to server

Usage:
    from app.fl.ecg_client import ECGFLClient
    
    client = ECGFLClient(
        client_id=1,
        hospital_dir='ecg_dataset/hospital_1'
    )
    
    # Train on local data
    weights, n_samples, metrics = client.train(global_weights)
"""

import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Subset
from typing import Tuple, Dict, List
from sklearn.model_selection import train_test_split

from app.fl.ecg_model import ECGModel
from app.fl.ecg_unified import ECGUnifiedPipeline


class ECGLocalDataset(Dataset):
    """
    PyTorch Dataset for hospital-local ECG data.
    
    Loads WFDB files on-demand with unified preprocessing.
    """
    
    def __init__(self, file_paths: List[str], labels: np.ndarray):
        """
        Args:
            file_paths: List of WFDB file paths (without extension)
            labels: Array of binary labels (0=Normal, 1=Arrhythmia)
        """
        self.file_paths = file_paths
        self.labels = labels
        self.pipeline = ECGUnifiedPipeline()
    
    def __len__(self):
        return len(self.file_paths)
    
    def __getitem__(self, idx):
        """
        Load and preprocess single ECG sample.
        
        Returns:
            Tuple of (signal_12lead, label)
            signal_12lead: (12, 1000) tensor
            label: scalar tensor
        """
        filepath = self.file_paths[idx]
        label = self.labels[idx]
        
        # Load and preprocess
        signal_12lead, _ = self.pipeline.load_and_preprocess(filepath)
        
        # Convert to tensor and transpose to (12, 1000)
        signal_tensor = torch.tensor(signal_12lead, dtype=torch.float32).transpose(0, 1)
        label_tensor = torch.tensor(label, dtype=torch.long)
        
        return signal_tensor, label_tensor


class ECGFLClient:
    """
    Federated Learning client for ECG model training.
    
    Runs on hospital nodes. Never shares raw ECG data.
    """
    
    def __init__(self, client_id: int, hospital_dir: str):
        """
        Initialize ECG FL client.
        
        Args:
            client_id: Unique hospital identifier (1, 2, or 3)
            hospital_dir: Path to hospital dataset directory
        """
        self.client_id = client_id
        self.hospital_dir = hospital_dir
        self.model = ECGModel(num_classes=2)  # Binary classification
        
        # Load dataset
        self.pipeline = ECGUnifiedPipeline()
        self.file_paths, self.labels = self.pipeline.load_hospital_dataset(hospital_dir)
        
        # --- Dataset Validation ---
        n_total = len(self.file_paths)
        n_normal = int(sum(self.labels == 0))
        n_arrhythmia = int(sum(self.labels == 1))
        print(f"ECG Client {client_id}: Loaded {n_total} local samples")
        print(f"  Normal: {n_normal}, Arrhythmia: {n_arrhythmia}")
        
        if n_total == 0:
            raise ValueError(f"Client {client_id}: No samples found in {hospital_dir}")
        
        if n_normal == 0:
            print(f"  WARNING: Client {client_id} has NO Normal samples — model will be biased")
        if n_arrhythmia == 0:
            print(f"  WARNING: Client {client_id} has NO Arrhythmia samples — model will be biased")
        
        ratio = n_arrhythmia / n_total if n_total > 0 else 0
        print(f"  Arrhythmia ratio: {ratio:.2%}")
        if ratio < 0.1 or ratio > 0.9:
            print(f"  WARNING: Severe class imbalance detected (ratio={ratio:.2%}). Class weighting will be applied.")
    
    def update_model(self, global_weights: Dict):
        """Update local model with global weights from server."""
        if global_weights:
            self.model.set_weights(global_weights)
    
    def train(
        self,
        global_weights: Dict,
        epochs: int = 5,
        batch_size: int = 16,
        lr: float = 0.0003
    ) -> Tuple[Dict, int, Dict]:
        """
        Train local model on hospital's private ECG data.
        
        Args:
            global_weights: Global model weights from server
            epochs: Number of training epochs
            batch_size: Batch size for training
            lr: Learning rate
            
        Returns:
            Tuple of (trained_weights, n_samples, metrics)
            - trained_weights: Updated model weights
            - n_samples: Number of samples trained on
            - metrics: Dict with 'accuracy', 'loss'
        """
        import copy
        
        # Check if dataset is empty
        if len(self.file_paths) == 0:
            raise ValueError(f"No ECG samples found in hospital directory: {self.hospital_dir}. Please ensure metadata.csv and WFDB files (.dat, .hea) exist.")
        
        # Update local model with global weights
        self.update_model(global_weights)
        
        # Save pre-training weights for differential privacy
        pre_train_weights = copy.deepcopy(self.model.get_weights()) if global_weights else None
        
        # --- Stratified Train/Validation Split (70% train, 30% val) ---
        n_total = len(self.file_paths)
        unique_labels = np.unique(self.labels)
        val_dataloader = None
        
        if n_total >= 10 and len(unique_labels) > 1:
            # Stratified split to preserve class ratio
            indices = np.arange(n_total)
            train_idx, val_idx = train_test_split(
                indices,
                test_size=0.3,
                stratify=self.labels,
                random_state=42
            )
            
            full_dataset = ECGLocalDataset(self.file_paths, self.labels)
            train_dataset = Subset(full_dataset, train_idx)
            val_dataset = Subset(full_dataset, val_idx)
            
            train_labels = self.labels[train_idx]
            print(f"  Train split: {len(train_idx)} samples | Val split: {len(val_idx)} samples")
            print(f"  Train Normal: {sum(train_labels==0)}, Arrhythmia: {sum(train_labels==1)}")
        else:
            # Too small or single class — use all for training, no val split
            full_dataset = ECGLocalDataset(self.file_paths, self.labels)
            train_dataset = full_dataset
            train_labels = self.labels
            print(f"  Dataset too small ({n_total}) or single class — skipping val split")
        
        # --- Class Weighting (handle imbalance) ---
        n_normal = int(sum(train_labels == 0))
        n_arrhy = int(sum(train_labels == 1))
        if n_normal > 0 and n_arrhy > 0:
            total = n_normal + n_arrhy
            w_normal = total / (2.0 * n_normal)
            w_arrhy = total / (2.0 * n_arrhy)
            class_weights = torch.tensor([w_normal, w_arrhy], dtype=torch.float32)
            print(f"  Class weights — Normal: {w_normal:.3f}, Arrhythmia: {w_arrhy:.3f}")
        else:
            class_weights = None
        
        # Create dataloaders
        eff_batch = min(batch_size, len(train_dataset))
        dataloader = DataLoader(train_dataset, batch_size=eff_batch, shuffle=True, num_workers=0)
        if val_dataloader is None and n_total >= 10 and len(unique_labels) > 1:
            val_dataloader = DataLoader(val_dataset, batch_size=eff_batch, shuffle=False, num_workers=0)
        
        # Train with validation + early stopping
        metrics = self.model.train_local(
            dataloader,
            epochs=epochs,
            lr=lr,
            val_dataloader=val_dataloader,
            class_weights=class_weights,
            early_stopping_patience=3
        )
        
        # Log final metrics
        print(f"  Final Train Loss: {metrics['loss']:.4f} | Acc: {metrics['accuracy']:.4f}")
        if metrics.get('val_accuracy') is not None:
            cm = metrics.get('confusion_matrix', [])
            print(f"  Final Val Loss: {metrics['val_loss']:.4f} | Acc: {metrics['val_accuracy']:.4f} | "
                  f"F1: {metrics.get('val_f1', 0):.4f}")
            if cm:
                print(f"  Confusion Matrix: TN={cm[0][0]} FP={cm[0][1]} FN={cm[1][0]} TP={cm[1][1]}")
                print(f"  Precision: {metrics.get('val_precision', 0):.4f} | "
                      f"Recall: {metrics.get('val_recall', 0):.4f}")
        
        # Get trained weights
        trained_weights = self.model.get_weights()
        # Use actual training examples (train split, not total file count)
        n_samples = len(train_dataset)
        
        # Differential Privacy: Clip + Noise on weight deltas
        if pre_train_weights is not None:
            max_delta_norm = 5.0
            noise_multiplier = 0.01
            
            for k in trained_weights.keys():
                # Compute weight update (delta)
                delta = trained_weights[k] - pre_train_weights[k]
                
                # Clip delta's L2 norm
                delta_norm = torch.norm(delta.float()).item()
                if delta_norm > max_delta_norm:
                    delta = delta * (max_delta_norm / delta_norm)
                
                # Add Gaussian noise (convert to float for noise generation)
                noise = torch.randn_like(delta.float()) * (noise_multiplier * max_delta_norm)
                delta = delta + noise.to(delta.dtype)
                
                # Reconstruct weight
                trained_weights[k] = pre_train_weights[k] + delta
        
        return trained_weights, n_samples, metrics


# Backward compatibility alias
ECGClient = ECGFLClient
