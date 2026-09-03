"""
PyTorch-based Neural Network (MLP) with GPU support.
"""
import numpy as np
from typing import Optional, Dict, Any, List
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import Dataset, DataLoader
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False

    def tqdm(iterable=None, **kwargs):
        return iterable if iterable is not None else range(0)

from .base import BaseModel
from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gc_utils import collect_after_operation, get_gc_manager

logger = get_logger(__name__)


class MLPDataset(Dataset):
    """Dataset for PyTorch DataLoader."""
    
    def __init__(self, X, y=None):
        """
        Initialize dataset.
        
        Args:
            X: Feature matrix
            y: Target values (optional)
        """
        if isinstance(X, csr_matrix):
            X = X.toarray()
        
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y) if y is not None else None
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        if self.y is not None:
            return self.X[idx], self.y[idx]
        return self.X[idx]


class MLP(nn.Module):
    """Multi-Layer Perceptron."""
    
    def __init__(self, input_dim: int, hidden_layers: List[int], dropout: float = 0.3):
        """
        Initialize MLP.
        
        Args:
            input_dim: Input dimension
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
        """
        super(MLP, self).__init__()
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Output layer (binary classification)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.model(x).squeeze()


class NeuralNetworkModel(BaseModel):
    """PyTorch-based Neural Network model."""
    
    def __init__(
        self,
        config=None,
        hidden_layers: Optional[List[int]] = None,
        dropout: float = 0.3,
        learning_rate: float = 0.001,
        batch_size: int = 32,
        epochs: int = 50,
        early_stopping_patience: int = 5,
        **kwargs
    ):
        """
        Initialize Neural Network model.
        
        Args:
            config: Configuration object
            hidden_layers: List of hidden layer sizes
            dropout: Dropout rate
            learning_rate: Learning rate
            batch_size: Batch size
            epochs: Number of epochs
            early_stopping_patience: Early stopping patience
            **kwargs: Additional arguments
        """
        super().__init__(config, model_name="neural_network")
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required for NeuralNetworkModel")
        
        self.hidden_layers = hidden_layers or (config.hidden_layers if config else [512, 256])
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.early_stopping_patience = early_stopping_patience
        self.kwargs = kwargs
        
        # Determine device
        self.device = torch.device('cuda' if torch.cuda.is_available() and self.use_gpu else 'cpu')
        self.logger.info(f"Using device: {self.device}")
        
        self.model = None
        self.optimizer = None
        self.criterion = nn.BCELoss()
        self.input_dim = None
    
    def fit(self, X, y, **kwargs):
        """Fit the model."""
        self.logger.info("Fitting Neural Network model")
        
        # Convert sparse to dense if needed
        if isinstance(X, csr_matrix):
            X = X.toarray()
            collect_after_operation("sparse_to_dense", aggressive=True)
        
        # Validate input shape
        if X.shape[0] == 0:
            raise ValueError("Cannot fit model: X is empty (0 samples)")
        if len(X.shape) < 2 or X.shape[1] == 0:
            raise ValueError(f"Cannot fit model: Invalid X shape {X.shape}")
        
        self.input_dim = X.shape[1]
        
        # Create model
        self.model = MLP(
            input_dim=self.input_dim,
            hidden_layers=self.hidden_layers,
            dropout=self.dropout
        ).to(self.device)
        
        # GC after model creation
        if self.device.type == 'cuda':
            collect_after_operation("model_to_gpu", aggressive=True)
        
        # Create optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.learning_rate)
        
        # Create dataset and dataloader
        dataset = MLPDataset(X, y)
        del X, y  # Free memory
        collect_after_operation("dataset_creation", aggressive=True)
        
        dataloader = DataLoader(
            dataset,
            batch_size=self.batch_size,
            shuffle=True
        )
        
        # Training loop with gradient accumulation
        best_loss = float('inf')
        patience_counter = 0
        accumulation_steps = getattr(self.config, 'gradient_accumulation_steps', 4) if hasattr(self, 'config') else 4
        
        epoch_iter = range(self.epochs)
        if TQDM_AVAILABLE:
            epoch_iter = tqdm(
                epoch_iter,
                desc="NN epochs",
                ascii=True,
                mininterval=30,
                leave=False
            )
        else:
            self.logger.warning("tqdm not available; epoch progress disabled")

        for epoch in epoch_iter:
            self.model.train()
            epoch_loss = 0.0
            self.optimizer.zero_grad()
            
            for batch_idx, (batch_X, batch_y) in enumerate(dataloader):
                batch_X = batch_X.to(self.device)
                batch_y = batch_y.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_X)
                loss = self.criterion(outputs, batch_y)
                loss = loss / accumulation_steps
                
                # Backward pass (accumulate gradients)
                loss.backward()
                
                # Update weights every accumulation_steps
                if (batch_idx + 1) % accumulation_steps == 0 or (batch_idx + 1) == len(dataloader):
                    self.optimizer.step()
                    self.optimizer.zero_grad()
                    
                    # HYPER-AGGRESSIVE GC after optimizer step (GPU intensive)
                    if self.device.type == 'cuda':
                        get_gc_manager().clear_caches()
                
                epoch_loss += loss.item() * accumulation_steps
                
                # Free batch tensors
                del batch_X, batch_y, outputs, loss
                
                # GC after every batch (HYPER-AGGRESSIVE)
                if (batch_idx + 1) % 10 == 0:  # Every 10 batches
                    collect_after_operation("batch_processing", aggressive=True)
            
            avg_loss = epoch_loss / len(dataloader)
            if TQDM_AVAILABLE and hasattr(epoch_iter, "set_postfix"):
                epoch_iter.set_postfix(loss=f"{avg_loss:.4f}")
            
            if (epoch + 1) % 10 == 0:
                self.logger.info(f"Epoch {epoch+1}/{self.epochs}, Loss: {avg_loss:.4f}")
            
            # HYPER-AGGRESSIVE GC after each epoch
            collect_after_operation(f"epoch_{epoch+1}", aggressive=True)
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.early_stopping_patience:
                    self.logger.info(f"Early stopping at epoch {epoch+1}")
                    break
        
        # Final GC after training
        collect_after_operation("training_complete", aggressive=True)
        
        self._fitted = True
        self.logger.info("Neural Network model fitted")
        return self
    
    def predict(self, X) -> np.ndarray:
        """Predict class labels."""
        self._ensure_fitted()
        
        if isinstance(X, csr_matrix):
            X = X.toarray()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            predictions = (outputs.cpu().numpy() > 0.5).astype(int)
            del X_tensor, outputs
        
        # GC after prediction
        if self.device.type == 'cuda':
            collect_after_operation("predict", aggressive=True)
        
        return predictions
    
    def predict_proba(self, X) -> np.ndarray:
        """Predict class probabilities."""
        self._ensure_fitted()
        
        if isinstance(X, csr_matrix):
            X = X.toarray()
        
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X).to(self.device)
            outputs = self.model(X_tensor)
            probabilities = outputs.cpu().numpy()
            del X_tensor, outputs
        
        # Return probabilities for both classes
        proba = np.zeros((len(probabilities), 2))
        proba[:, 1] = probabilities
        proba[:, 0] = 1 - probabilities
        del probabilities
        
        # GC after predict_proba
        if self.device.type == 'cuda':
            collect_after_operation("predict_proba", aggressive=True)
        
        return proba
    
    def save(self, path: str):
        """Save model to disk."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving Neural Network model to {path}")
        
        model_state = {
            'model_state_dict': self.model.state_dict() if self.model else None,
            'optimizer_state_dict': self.optimizer.state_dict() if self.optimizer else None,
            'input_dim': self.input_dim,
            'hidden_layers': self.hidden_layers,
            'dropout': self.dropout,
            'learning_rate': self.learning_rate,
            'batch_size': self.batch_size,
            'epochs': self.epochs,
            'early_stopping_patience': self.early_stopping_patience,
            'kwargs': self.kwargs,
            'fitted': self._fitted
        }
        
        with open(path, 'wb') as f:
            pickle.dump(model_state, f)
        
        self.logger.info("Model saved")
    
    @classmethod
    def load(cls, path: str, config=None):
        """Load model from disk."""
        logger.info(f"Loading Neural Network model from {path}")
        
        with open(path, 'rb') as f:
            model_state = pickle.load(f)
        
        model = cls(
            config=config,
            hidden_layers=model_state['hidden_layers'],
            dropout=model_state['dropout'],
            learning_rate=model_state['learning_rate'],
            batch_size=model_state['batch_size'],
            epochs=model_state['epochs'],
            early_stopping_patience=model_state['early_stopping_patience'],
            **model_state['kwargs']
        )
        
        if model_state['model_state_dict']:
            model.input_dim = model_state['input_dim']
            model.model = MLP(
                input_dim=model.input_dim,
                hidden_layers=model.hidden_layers,
                dropout=model.dropout
            ).to(model.device)
            model.model.load_state_dict(model_state['model_state_dict'])
        
        if model_state['optimizer_state_dict'] and model.optimizer:
            model.optimizer.load_state_dict(model_state['optimizer_state_dict'])
        
        model._fitted = model_state['fitted']
        
        logger.info("Model loaded")
        return model
