import os
import pandas as pd
from app.fl.model import FLModel
from app.fl.data import FLDataHandler

class FLClient:
    def __init__(self, client_id, data_path):
        self.client_id = client_id
        self.data_path = data_path
        self.model = FLModel()
        self.data_handler = FLDataHandler()
        self.X_train = None
        self.y_train = None
        self._load_data()

    def _load_data(self):
        # Load and preprocess data
        self.X_train, self.y_train = self.data_handler.load_data(self.data_path)
        print(f"Client {self.client_id}: Loaded {len(self.X_train)} samples.")

    def update_model(self, global_weights):
        # Update local model with global weights before training
        if global_weights:
            self.model.set_weights(global_weights)

    def train(self, global_weights, data_path=None):
        """
        Train local model on private data.
        """
        # Update local model with global weights
        self.update_model(global_weights)

        # Save a copy of global weights BEFORE training for DP delta computation
        import copy
        import torch
        pre_train_weights = copy.deepcopy(self.model.get_weights()) if global_weights else None

        # Load local data if a specific path is provided for this training round
        if data_path:
            file_path = data_path
            data_loader = FLDataHandler()
            X_train_round, y_train_round = data_loader.load_data(file_path)
            print(f"Client {self.client_id}: Loaded {len(X_train_round)} samples for this training round from {file_path}.")
        else:
            # Use data loaded during initialization (from uploaded file or default path)
            X_train_round, y_train_round = self.X_train, self.y_train
            print(f"Client {self.client_id}: Training on {len(X_train_round)} samples from {self.data_path}.")

        # Local training
        metrics = self.model.train(X_train_round, y_train_round, epochs=20)
        
        # Extract trained weights
        n_samples = len(self.X_train)
        trained_weights = self.model.get_weights()
        
        # Differential Privacy: Clip + Noise on WEIGHT DELTAS (not raw weights)
        # This preserves the global model and only bounds the per-client update.
        # Note: For formal (ε,δ)-DP, use a framework like Opacus with privacy accounting.
        max_delta_norm = 5.0      # Clipping bound per parameter delta
        noise_multiplier = 0.01   # σ relative to sensitivity

        if pre_train_weights is not None:
            for k in trained_weights.keys():
                # Compute the weight update (delta)
                delta = trained_weights[k] - pre_train_weights[k]
                
                # Step 1: Clip the delta's L2 norm
                delta_norm = torch.norm(delta.float()).item()
                if delta_norm > max_delta_norm:
                    delta = delta * (max_delta_norm / delta_norm)
                
                # Step 2: Add Gaussian noise proportional to the clipping bound
                noise = torch.randn_like(delta) * (noise_multiplier * max_delta_norm)
                delta = delta + noise
                
                # Reconstruct the weight: global + clipped_noisy_delta
                trained_weights[k] = pre_train_weights[k] + delta

        return trained_weights, n_samples, metrics
