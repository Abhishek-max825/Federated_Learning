import torch
import torch.nn as nn
import os

class FederatedDNN(nn.Module):
    def __init__(self, input_size=23):
        super(FederatedDNN, self).__init__()
        # 3-layer Neural Network architecture
        self.layer1 = nn.Linear(input_size, 64)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(0.1)
        self.layer2 = nn.Linear(64, 32)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(0.1)
        self.layer3 = nn.Linear(32, 1)
        # No sigmoid here — use BCEWithLogitsLoss for numerical stability

    def forward(self, x):
        x = self.dropout1(self.relu1(self.layer1(x)))
        x = self.dropout2(self.relu2(self.layer2(x)))
        x = self.layer3(x)  # Raw logits
        return x

class FLModel:
    def __init__(self, input_size=23):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = FederatedDNN(input_size=input_size).to(self.device)

    def train(self, X, y, epochs=5, batch_size=256, lr=0.003):
        """Train the model on local data using PyTorch standard loop."""
        import torch.optim as optim

        # Compute class weight to handle imbalance
        # pos_weight = num_negatives / num_positives
        n_pos = max((y == 1).sum(), 1)
        n_neg = max((y == 0).sum(), 1)
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(self.device)
        print(f"  Class distribution: {n_neg} negative, {n_pos} positive, pos_weight={pos_weight.item():.2f}")

        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        optimizer = optim.Adam(self.model.parameters(), lr=lr)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)

        X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
        y_tensor = torch.tensor(y, dtype=torch.float32).view(-1, 1).to(self.device)
        
        # Mini-batch training
        dataset = torch.utils.data.TensorDataset(X_tensor, y_tensor)
        loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.model.train()

        for epoch in range(epochs):
            epoch_loss = 0
            epoch_correct = 0
            epoch_total = 0

            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                logits = self.model(batch_X)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * batch_X.size(0)
                
                # Calculate accuracy using sigmoid on logits
                predictions = (torch.sigmoid(logits) >= 0.5).float()
                epoch_correct += (predictions == batch_y).sum().item()
                epoch_total += batch_y.size(0)
            
            epoch_avg_loss = epoch_loss / epoch_total
            epoch_accuracy = epoch_correct / epoch_total
            scheduler.step(epoch_avg_loss)
            print(f"  Epoch {epoch+1}/{epochs} - Loss: {epoch_avg_loss:.4f}, Accuracy: {epoch_accuracy:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")

        # Return final epoch metrics (post-training performance)
        return {'loss': epoch_avg_loss, 'accuracy': epoch_accuracy}

    def get_weights(self):
        """Extract the model state dict (weights and biases)."""
        # Convert tensors to CPU before sending/saving
        state_dict = self.model.state_dict()
        return {k: v.cpu() for k, v in state_dict.items()}

    def set_weights(self, weights):
        """Update the model with an aggregated state dict."""
        # Move tensors to the correct device
        weights_device = {k: v.to(self.device) for k, v in weights.items()}
        self.model.load_state_dict(weights_device)

    def predict(self, X):
        """Predict binary class (0 or 1)."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor).squeeze()
            probs = torch.sigmoid(logits)
            predictions = (probs >= 0.5).int()
            # Handle single sample case
            if predictions.dim() == 0:
                return [predictions.item()]
            return predictions.cpu().numpy()

    def predict_proba(self, X):
        """Predict probabilities for [No Disease, Disease]."""
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            logits = self.model(X_tensor).squeeze()
            prob_1 = torch.sigmoid(logits)
            if prob_1.dim() == 0:
                prob_1 = prob_1.item()
            else:
                prob_1 = prob_1.cpu().numpy()
            
            # Reconstruct scikit-learn style predict_proba: [prob_class_0, prob_class_1]
            import numpy as np
            prob_0 = 1.0 - prob_1
            return np.vstack((prob_0, prob_1)).T

    def save(self, filepath):
        """Save the PyTorch model state."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
        }, filepath)

    def load(self, filepath):
        """Load the PyTorch model state."""
        if os.path.exists(filepath):
            checkpoint = torch.load(filepath, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            return True
        return False
