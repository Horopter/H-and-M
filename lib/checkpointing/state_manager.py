"""
Training state management for tracking progress and experiment metadata.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class StateManager:
    """Manage training state and experiment metadata."""
    
    def __init__(self, config=None, state_dir: Optional[str] = None):
        """
        Initialize state manager.
        
        Args:
            config: Configuration object
            state_dir: State directory (uses checkpoint_dir if None)
        """
        self.config = config or get_config()
        self.state_dir = Path(state_dir or self.config.checkpoint_dir) / "states"
        self.state_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
    
    def create_experiment(
        self,
        experiment_name: str,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Create a new experiment.
        
        Args:
            experiment_name: Name of the experiment
            metadata: Experiment metadata
            
        Returns:
            Path to experiment directory
        """
        experiment_dir = self.state_dir / experiment_name
        experiment_dir.mkdir(parents=True, exist_ok=True)
        
        experiment_metadata = {
            'name': experiment_name,
            'created_at': datetime.now().isoformat(),
            'config': self.config.to_dict(),
            **(metadata or {})
        }
        
        metadata_path = experiment_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(experiment_metadata, f, indent=2, default=str)
        
        self.logger.info(f"Created experiment: {experiment_name}")
        return experiment_dir
    
    def save_training_progress(
        self,
        experiment_name: str,
        model_type: str,
        fold_id: Optional[int],
        epoch: int,
        metrics: Dict[str, float],
        loss: Optional[float] = None
    ):
        """
        Save training progress.
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model
            fold_id: Fold ID (optional)
            epoch: Epoch number
            metrics: Dictionary of metrics
            loss: Training loss (optional)
        """
        experiment_dir = self.state_dir / experiment_name
        
        progress_file = experiment_dir / f"{model_type}_progress.json"
        
        # Load existing progress
        if progress_file.exists():
            with open(progress_file, 'r') as f:
                progress = json.load(f)
        else:
            progress = {}
        
        # Update progress
        key = f"fold_{fold_id}" if fold_id is not None else "no_fold"
        if key not in progress:
            progress[key] = []
        
        progress_entry = {
            'epoch': epoch,
            'metrics': metrics,
            'timestamp': datetime.now().isoformat()
        }
        if loss is not None:
            progress_entry['loss'] = loss
        
        progress[key].append(progress_entry)
        
        # Save progress
        with open(progress_file, 'w') as f:
            json.dump(progress, f, indent=2, default=str)
        
        self.logger.debug(f"Saved training progress for {model_type}, fold {fold_id}, epoch {epoch}")
    
    def get_training_progress(
        self,
        experiment_name: str,
        model_type: str,
        fold_id: Optional[int] = None
    ) -> list:
        """
        Get training progress.
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model
            fold_id: Fold ID (optional)
            
        Returns:
            List of progress entries
        """
        experiment_dir = self.state_dir / experiment_name
        progress_file = experiment_dir / f"{model_type}_progress.json"
        
        if not progress_file.exists():
            return []
        
        with open(progress_file, 'r') as f:
            progress = json.load(f)
        
        key = f"fold_{fold_id}" if fold_id is not None else "no_fold"
        return progress.get(key, [])
    
    def save_best_metrics(
        self,
        experiment_name: str,
        model_type: str,
        fold_id: Optional[int],
        metrics: Dict[str, float],
        hyperparams: Optional[Dict[str, Any]] = None
    ):
        """
        Save best metrics for a model.
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model
            fold_id: Fold ID (optional)
            metrics: Dictionary of best metrics
            hyperparams: Best hyperparameters (optional)
        """
        experiment_dir = self.state_dir / experiment_name
        best_metrics_file = experiment_dir / f"{model_type}_best_metrics.json"
        
        # Load existing best metrics
        if best_metrics_file.exists():
            with open(best_metrics_file, 'r') as f:
                best_metrics = json.load(f)
        else:
            best_metrics = {}
        
        # Update best metrics
        key = f"fold_{fold_id}" if fold_id is not None else "no_fold"
        best_metrics[key] = {
            'metrics': metrics,
            'hyperparams': hyperparams,
            'timestamp': datetime.now().isoformat()
        }
        
        # Save best metrics
        with open(best_metrics_file, 'w') as f:
            json.dump(best_metrics, f, indent=2, default=str)
        
        self.logger.info(f"Saved best metrics for {model_type}, fold {fold_id}")
    
    def get_best_metrics(
        self,
        experiment_name: str,
        model_type: str,
        fold_id: Optional[int] = None
    ) -> Optional[Dict[str, Any]]:
        """
        Get best metrics for a model.
        
        Args:
            experiment_name: Name of the experiment
            model_type: Type of model
            fold_id: Fold ID (optional)
            
        Returns:
            Dictionary of best metrics or None
        """
        experiment_dir = self.state_dir / experiment_name
        best_metrics_file = experiment_dir / f"{model_type}_best_metrics.json"
        
        if not best_metrics_file.exists():
            return None
        
        with open(best_metrics_file, 'r') as f:
            best_metrics = json.load(f)
        
        key = f"fold_{fold_id}" if fold_id is not None else "no_fold"
        return best_metrics.get(key)

