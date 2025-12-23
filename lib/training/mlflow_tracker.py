"""
MLFlow integration for experiment tracking and visualization.
"""
import os
from typing import Dict, Any, Optional, List
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class MLFlowTracker:
    """MLFlow experiment tracker."""
    
    def __init__(self, config=None, experiment_name: str = "kaggle_ml_pipeline"):
        """
        Initialize MLFlow tracker.
        
        Args:
            config: Configuration object
            experiment_name: Name of the MLFlow experiment
        """
        self.config = config or get_config()
        self.experiment_name = experiment_name
        self.logger = get_logger(self.__class__.__name__)
        
        if not MLFLOW_AVAILABLE:
            self.logger.warning("MLFlow not available, tracking disabled")
            self.enabled = False
            return
        
        self.enabled = True
        mlflow.set_experiment(experiment_name)
        self.logger.info(f"MLFlow experiment: {experiment_name}")
    
    def start_run(self, run_name: Optional[str] = None):
        """Start a new MLFlow run."""
        if not self.enabled:
            return None
        
        return mlflow.start_run(run_name=run_name)
    
    def end_run(self):
        """End current MLFlow run."""
        if self.enabled:
            mlflow.end_run()
    
    def log_params(self, params: Dict[str, Any]):
        """Log parameters."""
        if self.enabled:
            mlflow.log_params(params)
    
    def log_metrics(self, metrics: Dict[str, float], step: Optional[int] = None):
        """Log metrics."""
        if self.enabled:
            mlflow.log_metrics(metrics, step=step)
    
    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if self.enabled:
            mlflow.log_metric(key, value, step=step)
    
    def log_artifacts(self, local_dir: str, artifact_path: Optional[str] = None):
        """Log artifacts."""
        if self.enabled:
            mlflow.log_artifacts(local_dir, artifact_path)
    
    def log_figure(self, figure, artifact_file: str):
        """Log a matplotlib figure."""
        if self.enabled:
            mlflow.log_figure(figure, artifact_file)
    
    def log_model(self, model, artifact_path: str = "model"):
        """Log a model."""
        if self.enabled:
            try:
                mlflow.sklearn.log_model(model, artifact_path)
            except Exception as e:
                self.logger.warning(f"Could not log model to MLFlow: {e}")
    
    def plot_training_curves(
        self,
        train_losses: List[float],
        val_losses: List[float],
        train_metrics: Dict[str, List[float]],
        val_metrics: Dict[str, List[float]],
        save_path: Optional[str] = None
    ):
        """Create and log training curves."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Training Progress', fontsize=16)
        
        epochs = range(1, len(train_losses) + 1)
        
        # Loss curves
        axes[0, 0].plot(epochs, train_losses, 'b-', label='Train Loss')
        axes[0, 0].plot(epochs, val_losses, 'r-', label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Loss Curves')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # F1 Score
        if 'f1' in train_metrics and 'f1' in val_metrics:
            axes[0, 1].plot(epochs, train_metrics['f1'], 'b-', label='Train F1')
            axes[0, 1].plot(epochs, val_metrics['f1'], 'r-', label='Val F1')
            axes[0, 1].set_xlabel('Epoch')
            axes[0, 1].set_ylabel('F1 Score')
            axes[0, 1].set_title('F1 Score')
            axes[0, 1].legend()
            axes[0, 1].grid(True)
        
        # Accuracy
        if 'accuracy' in train_metrics and 'accuracy' in val_metrics:
            axes[0, 2].plot(epochs, train_metrics['accuracy'], 'b-', label='Train Acc')
            axes[0, 2].plot(epochs, val_metrics['accuracy'], 'r-', label='Val Acc')
            axes[0, 2].set_xlabel('Epoch')
            axes[0, 2].set_ylabel('Accuracy')
            axes[0, 2].set_title('Accuracy')
            axes[0, 2].legend()
            axes[0, 2].grid(True)
        
        # Precision
        if 'precision' in train_metrics and 'precision' in val_metrics:
            axes[1, 0].plot(epochs, train_metrics['precision'], 'b-', label='Train Prec')
            axes[1, 0].plot(epochs, val_metrics['precision'], 'r-', label='Val Prec')
            axes[1, 0].set_xlabel('Epoch')
            axes[1, 0].set_ylabel('Precision')
            axes[1, 0].set_title('Precision')
            axes[1, 0].legend()
            axes[1, 0].grid(True)
        
        # Recall
        if 'recall' in train_metrics and 'recall' in val_metrics:
            axes[1, 1].plot(epochs, train_metrics['recall'], 'b-', label='Train Rec')
            axes[1, 1].plot(epochs, val_metrics['recall'], 'r-', label='Val Rec')
            axes[1, 1].set_xlabel('Epoch')
            axes[1, 1].set_ylabel('Recall')
            axes[1, 1].set_title('Recall')
            axes[1, 1].legend()
            axes[1, 1].grid(True)
        
        # ROC-AUC
        if 'roc_auc' in train_metrics and 'roc_auc' in val_metrics:
            axes[1, 2].plot(epochs, train_metrics['roc_auc'], 'b-', label='Train AUC')
            axes[1, 2].plot(epochs, val_metrics['roc_auc'], 'r-', label='Val AUC')
            axes[1, 2].set_xlabel('Epoch')
            axes[1, 2].set_ylabel('ROC-AUC')
            axes[1, 2].set_title('ROC-AUC')
            axes[1, 2].legend()
            axes[1, 2].grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        if self.enabled:
            self.log_figure(fig, "training_curves.png")
        
        plt.close(fig)
        return fig

