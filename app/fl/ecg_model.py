"""
ResNet1D-18 Architecture for ECG Classification.

Architecture:
- 18-layer residual CNN optimized for 1D time-series
- BatchNorm for stable training in FL
- Attention mechanism for interpretability
- Binary classification: Normal vs Arrhythmia

Input: (batch, 12, 1000) - 12 leads, 1000 timesteps
Output: (batch, 2) - [Normal, Arrhythmia] probabilities
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple


class BasicBlock1D(nn.Module):
    """Basic residual block for 1D CNN."""
    
    expansion = 1
    
    def __init__(
        self, 
        in_channels: int, 
        out_channels: int, 
        stride: int = 1,
        downsample: nn.Module = None
    ):
        super(BasicBlock1D, self).__init__()
        
        # First conv layer
        self.conv1 = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=7, 
            stride=stride,
            padding=3,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        # Second conv layer
        self.conv2 = nn.Conv1d(
            out_channels, 
            out_channels, 
            kernel_size=7,
            stride=1,
            padding=3,
            bias=False
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with residual connection."""
        identity = x
        
        # First conv block
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        # Second conv block
        out = self.conv2(out)
        out = self.bn2(out)
        
        # Downsampling for residual if needed
        if self.downsample is not None:
            identity = self.downsample(x)
        
        # Residual connection
        out += identity
        out = self.relu(out)
        
        return out


class ResNet1D(nn.Module):
    """ResNet1D architecture for ECG classification."""
    
    def __init__(
        self,
        block: nn.Module,
        layers: list,
        num_classes: int = 2,
        in_channels: int = 12,
        attention: bool = True
    ):
        super(ResNet1D, self).__init__()
        
        self.in_channels = 64
        self.attention = attention
        
        # Initial conv layer
        self.conv1 = nn.Conv1d(
            in_channels, 
            64, 
            kernel_size=15, 
            stride=2, 
            padding=7,
            bias=False
        )
        self.bn1 = nn.BatchNorm1d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Residual layers
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # Attention mechanism
        if self.attention:
            self.attention_conv = nn.Conv1d(512, 1, kernel_size=1)
        
        # Global average pooling and classifier
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(p=0.4)
        self.fc = nn.Linear(512 * block.expansion, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _make_layer(
        self, 
        block: nn.Module, 
        out_channels: int, 
        blocks: int, 
        stride: int = 1
    ) -> nn.Sequential:
        """Create a layer with multiple residual blocks."""
        downsample = None
        if stride != 1 or self.in_channels != out_channels * block.expansion:
            downsample = nn.Sequential(
                nn.Conv1d(
                    self.in_channels,
                    out_channels * block.expansion,
                    kernel_size=1,
                    stride=stride,
                    bias=False
                ),
                nn.BatchNorm1d(out_channels * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels * block.expansion
        
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        
        return nn.Sequential(*layers)
    
    def _initialize_weights(self):
        """Initialize weights using Kaiming initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.zeros_(m.bias)  # Zero bias → equal logits → ~50/50 before training
    
    def forward(
        self, 
        x: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor (batch, 12, 1000)
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of (output, attention_map)
        """
        # Initial conv
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        # Residual blocks
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Attention mechanism
        attention_map = None
        if self.attention:
            raw_attention = self.attention_conv(x)   # (B, 1, T)
            attention_weights = torch.sigmoid(raw_attention)  # 0-1 weights
            x = x * attention_weights  # Apply attention to features
            
            if return_attention:
                attention_map = F.interpolate(
                    attention_weights,
                    size=1000,
                    mode='linear',
                    align_corners=False
                )
        
        # Global pooling
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        
        # Classification with dropout
        x = self.dropout(x)
        x = self.fc(x)
        
        return x, attention_map


def resnet1d_18(num_classes: int = 2, in_channels: int = 12, attention: bool = True) -> ResNet1D:
    """
    Create ResNet1D-18 model for binary ECG classification.
    
    Args:
        num_classes: Number of output classes (default 2: Normal=0, Arrhythmia=1)
        in_channels: Number of input channels (default 12 for 12-lead ECG)
        attention: Whether to use attention mechanism
        
    Returns:
        ResNet1D-18 model
    """
    assert num_classes == 2, "Binary classification only - num_classes must be 2"
    
    return ResNet1D(
        block=BasicBlock1D,
        layers=[2, 2, 2, 2],  # 2 blocks per layer = 18 total layers
        num_classes=num_classes,
        in_channels=in_channels,
        attention=attention
    )


class ECGModel:
    """
    Wrapper class for ECG classification model.
    
    Handles:
    - Model initialization
    - Training (both batch and dataloader-based)
    - Prediction with confidence scores
    - Weight extraction for FL
    
    NOTE: Binary classification (Normal vs Arrhythmia)
    """
    
    def __init__(
        self,
        num_classes: int = 2,
        in_channels: int = 12,
        attention: bool = True
    ):
        """
        Initialize ECG model.
        
        Args:
            num_classes: Number of classes (default 2 for binary)
            in_channels: Number of input leads (default 12)
            attention: Use attention mechanism
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = resnet1d_18(
            num_classes=num_classes,
            in_channels=in_channels,
            attention=attention
        ).to(self.device)
        
        self.num_classes = num_classes
        assert self.num_classes == 2, "ECGModel only supports binary classification (Normal/Arrhythmia)"
    
    def train_local(
        self,
        dataloader: DataLoader,
        epochs: int = 5,
        lr: float = 0.001,
        val_dataloader: DataLoader = None,
        class_weights: torch.Tensor = None,
        early_stopping_patience: int = 3
    ) -> dict:
        """
        Train model using a PyTorch DataLoader.
        
        Used by ECGFLClient for federated learning.
        
        Args:
            dataloader: PyTorch DataLoader with (signal, label) batches
            epochs: Number of training epochs
            lr: Learning rate
            val_dataloader: Optional validation DataLoader
            class_weights: Optional class weights for imbalanced data
            early_stopping_patience: Stop if val_loss doesn't improve for N epochs
            
        Returns:
            Dictionary with training and validation metrics
        """
        import torch.optim as optim
        
        # Use class-weighted loss with label smoothing (prevents 100% confidence collapse)
        criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None,
            label_smoothing=0.1
        )
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        best_weights = None
        
        final_metrics = {'loss': 0.0, 'accuracy': 0.0, 'val_loss': None, 'val_accuracy': None}
        
        for epoch in range(epochs):
            # --- Training phase ---
            self.model.train()
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0
            
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                optimizer.zero_grad()
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                optimizer.step()
                
                epoch_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                epoch_correct += (predicted == batch_y).sum().item()
                epoch_total += batch_y.size(0)
            
            train_loss = epoch_loss / epoch_total if epoch_total > 0 else 0
            train_acc = epoch_correct / epoch_total if epoch_total > 0 else 0
            scheduler.step(train_loss)
            
            # --- Validation phase ---
            if val_dataloader is not None:
                val_loss, val_acc, val_metrics = self._evaluate(val_dataloader, criterion)
                final_metrics['val_loss'] = val_loss
                final_metrics['val_accuracy'] = val_acc
                final_metrics['val_precision'] = val_metrics.get('precision', 0.0)
                final_metrics['val_recall'] = val_metrics.get('recall', 0.0)
                final_metrics['val_f1'] = val_metrics.get('f1', 0.0)
                final_metrics['confusion_matrix'] = val_metrics.get('confusion_matrix', [])
                
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
                      f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
                      f"F1: {val_metrics.get('f1', 0.0):.4f}")
                
                # Early stopping on validation loss
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    patience_counter = 0
                    import copy
                    best_weights = copy.deepcopy(self.model.state_dict())
                else:
                    patience_counter += 1
                    if patience_counter >= early_stopping_patience:
                        print(f"  Early stopping at epoch {epoch+1} (val_loss no improvement for {early_stopping_patience} epochs)")
                        break
            else:
                print(f"  Epoch {epoch+1}/{epochs} | "
                      f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}")
            
            final_metrics['loss'] = train_loss
            final_metrics['accuracy'] = train_acc
        
        # Restore best weights if we have them
        if best_weights is not None:
            self.model.load_state_dict(best_weights)
        
        return final_metrics
    
    def _evaluate(self, dataloader: DataLoader, criterion) -> tuple:
        """Evaluate model on a dataloader. Returns (loss, accuracy, metrics dict)."""
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch_X, batch_y in dataloader:
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                outputs, _ = self.model(batch_X)
                loss = criterion(outputs, batch_y)
                total_loss += loss.item() * batch_X.size(0)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(batch_y.cpu().numpy())
        
        n = len(all_labels)
        avg_loss = total_loss / n if n > 0 else 0
        accuracy = sum(p == l for p, l in zip(all_preds, all_labels)) / n if n > 0 else 0
        
        # Confusion matrix and metrics
        tp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 1)
        fp = sum(1 for p, l in zip(all_preds, all_labels) if p == 1 and l == 0)
        tn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 0)
        fn = sum(1 for p, l in zip(all_preds, all_labels) if p == 0 and l == 1)
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        metrics = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'confusion_matrix': [[tn, fp], [fn, tp]]
        }
        
        return avg_loss, accuracy, metrics
    
    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        epochs: int = 10,
        batch_size: int = 32,
        lr: float = 0.001
    ) -> dict:
        """
        Train model on local ECG data (numpy arrays).
        
        Args:
            X: ECG data (num_samples, 12, 1000)
            y: Labels (num_samples,) - binary 0 or 1
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Learning rate
            
        Returns:
            Dictionary with training metrics
        """
        from torch.utils.data import TensorDataset, DataLoader
        
        # Prepare data
        X_tensor = torch.tensor(X, dtype=torch.float32)
        y_tensor = torch.tensor(y, dtype=torch.long)
        
        # Create dataset and dataloader
        dataset = TensorDataset(X_tensor, y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Train using the DataLoader method
        return self.train_local(dataloader, epochs=epochs, lr=lr)
    
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict classes and get probabilities.
        
        Args:
            X: ECG data (num_samples, 12, 1000)
            
        Returns:
            Tuple of (predictions, probabilities)
        """
        self.model.eval()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        TEMPERATURE = 2.0  # Softens overconfident predictions
        with torch.no_grad():
            outputs, _ = self.model(X_tensor)
            probabilities = F.softmax(outputs / TEMPERATURE, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        return predictions.cpu().numpy(), probabilities.cpu().numpy()
    
    def predict_with_attention(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict with attention maps for visualization.
        
        Args:
            X: ECG data (num_samples, 12, 1000)
            
        Returns:
            Tuple of (predictions, probabilities, attention_maps)
        """
        self.model.eval()
        
        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        
        TEMPERATURE = 2.0  # Softens overconfident predictions
        with torch.no_grad():
            outputs, attention_maps = self.model(X_tensor, return_attention=True)
            probabilities = F.softmax(outputs / TEMPERATURE, dim=1)
            _, predictions = torch.max(outputs, 1)
        
        return (
            predictions.cpu().numpy(),
            probabilities.cpu().numpy(),
            attention_maps.cpu().numpy() if attention_maps is not None else None
        )
    
    def get_weights(self) -> dict:
        """Extract model weights for FL aggregation."""
        state_dict = self.model.state_dict()
        return {k: v.cpu() for k, v in state_dict.items()}
    
    def set_weights(self, weights: dict):
        """Load aggregated weights."""
        weights_device = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(weights_device)
    
    def save(self, filepath: str):
        """Save model to file."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'num_classes': self.num_classes,
        }, filepath)
    
    def load(self, filepath: str) -> bool:
        """Load model from file."""
        import os
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return True
        return False
