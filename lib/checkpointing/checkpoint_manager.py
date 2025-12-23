"""
Checkpoint manager for saving model weights and training state in Arrow format.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime

from ..config import get_config
from ..logging.logger import get_logger
from ..utils.arrow_utils import ArrowUtils

logger = get_logger(__name__)


class CheckpointManager:
    """Manage model checkpoints with Arrow format."""
    
    def __init__(self, config=None, checkpoint_dir: Optional[str] = None):
        """
        Initialize checkpoint manager.
        
        Args:
            config: Configuration object
            checkpoint_dir: Checkpoint directory (uses config default if None)
        """
        self.config = config or get_config()
        self.checkpoint_dir = Path(checkpoint_dir or self.config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.logger = get_logger(self.__class__.__name__)
    
    def get_checkpoint_path(
        self,
        model_type: str,
        fold_id: Optional[int] = None,
        epoch: Optional[int] = None,
        score: Optional[float] = None
    ) -> Path:
        """
        Get checkpoint path for a model.
        
        Args:
            model_type: Type of model
            fold_id: Fold ID (optional)
            epoch: Epoch number (optional)
            score: Score (optional)
            
        Returns:
            Path to checkpoint
        """
        if fold_id is not None:
            fold_dir = self.checkpoint_dir / model_type / f"fold_{fold_id}"
            fold_dir.mkdir(parents=True, exist_ok=True)
            
            if epoch is not None and score is not None:
                filename = f"epoch_{epoch}_f1_{score:.4f}.arrow"
            elif epoch is not None:
                filename = f"epoch_{epoch}.arrow"
            else:
                filename = "model.arrow"
            
            return fold_dir / filename
        else:
            model_dir = self.checkpoint_dir / model_type
            model_dir.mkdir(parents=True, exist_ok=True)
            return model_dir / "model.arrow"
    
    def save_checkpoint(
        self,
        model,
        model_type: str,
        fold_id: Optional[int] = None,
        epoch: Optional[int] = None,
        score: Optional[float] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> Path:
        """
        Save model checkpoint.
        
        Args:
            model: Model instance
            model_type: Type of model
            fold_id: Fold ID (optional)
            epoch: Epoch number (optional)
            score: Score (optional)
            metadata: Additional metadata
            
        Returns:
            Path to saved checkpoint
        """
        checkpoint_path = self.get_checkpoint_path(model_type, fold_id, epoch, score)
        
        self.logger.info(f"Saving checkpoint to {checkpoint_path}")
        
        # Save model using model's save method
        model.save(str(checkpoint_path))
        
        # Save metadata
        if metadata:
            metadata_path = checkpoint_path.parent / "metadata.json"
            with open(metadata_path, 'w') as f:
                json.dump(metadata, f, indent=2, default=str)
        
        return checkpoint_path
    
    def load_checkpoint(
        self,
        model_class,
        checkpoint_path: str,
        config=None
    ):
        """
        Load model checkpoint.
        
        Args:
            model_class: Model class to load
            checkpoint_path: Path to checkpoint
            config: Configuration object
            
        Returns:
            Loaded model instance
        """
        self.logger.info(f"Loading checkpoint from {checkpoint_path}")
        
        model = model_class.load(checkpoint_path, config=config)
        
        return model
    
    def save_training_state(
        self,
        state: Dict[str, Any],
        model_type: str,
        fold_id: Optional[int] = None
    ) -> Path:
        """
        Save training state.
        
        Args:
            state: Training state dictionary
            model_type: Type of model
            fold_id: Fold ID (optional)
            
        Returns:
            Path to saved state
        """
        if fold_id is not None:
            state_path = self.checkpoint_dir / model_type / f"fold_{fold_id}" / "training_state.arrow"
        else:
            state_path = self.checkpoint_dir / model_type / "training_state.arrow"
        
        state_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving training state to {state_path}")
        
        # Save state using Arrow format
        ArrowUtils.save_dict(state, str(state_path))
        
        return state_path
    
    def load_training_state(
        self,
        model_type: str,
        fold_id: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Load training state.
        
        Args:
            model_type: Type of model
            fold_id: Fold ID (optional)
            
        Returns:
            Training state dictionary
        """
        if fold_id is not None:
            state_path = self.checkpoint_dir / model_type / f"fold_{fold_id}" / "training_state.arrow"
        else:
            state_path = self.checkpoint_dir / model_type / "training_state.arrow"
        
        if not state_path.exists():
            self.logger.warning(f"Training state not found at {state_path}")
            return {}
        
        self.logger.info(f"Loading training state from {state_path}")
        
        state = ArrowUtils.load_dict(str(state_path))
        return state
    
    def save_hyperparameters(
        self,
        hyperparams: Dict[str, Any],
        model_type: str,
        fold_id: Optional[int] = None
    ) -> Path:
        """
        Save hyperparameters.
        
        Args:
            hyperparams: Hyperparameter dictionary
            model_type: Type of model
            fold_id: Fold ID (optional)
            
        Returns:
            Path to saved hyperparameters
        """
        if fold_id is not None:
            hyperparams_path = self.checkpoint_dir / model_type / f"fold_{fold_id}" / "hyperparams.json"
        else:
            hyperparams_path = self.checkpoint_dir / model_type / "hyperparams.json"
        
        hyperparams_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.logger.info(f"Saving hyperparameters to {hyperparams_path}")
        
        with open(hyperparams_path, 'w') as f:
            json.dump(hyperparams, f, indent=2, default=str)
        
        return hyperparams_path
    
    def list_checkpoints(
        self,
        model_type: str,
        fold_id: Optional[int] = None
    ) -> list:
        """
        List available checkpoints.
        
        Args:
            model_type: Type of model
            fold_id: Fold ID (optional)
            
        Returns:
            List of checkpoint paths
        """
        if fold_id is not None:
            checkpoint_dir = self.checkpoint_dir / model_type / f"fold_{fold_id}"
        else:
            checkpoint_dir = self.checkpoint_dir / model_type
        
        if not checkpoint_dir.exists():
            return []
        
        checkpoints = list(checkpoint_dir.glob("*.arrow"))
        checkpoints.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        
        return checkpoints
    
    def get_best_checkpoint(
        self,
        model_type: str,
        fold_id: Optional[int] = None,
        metric: str = 'f1'
    ) -> Optional[Path]:
        """
        Get best checkpoint based on metric in filename.
        
        Args:
            model_type: Type of model
            fold_id: Fold ID (optional)
            metric: Metric name (default: 'f1')
            
        Returns:
            Path to best checkpoint or None
        """
        checkpoints = self.list_checkpoints(model_type, fold_id)
        
        if not checkpoints:
            return None
        
        # Extract scores from filenames
        best_checkpoint = None
        best_score = -1.0
        
        for checkpoint in checkpoints:
            # Try to extract score from filename (e.g., "epoch_10_f1_0.8500.arrow")
            import re
            match = re.search(rf'{metric}_([\d.]+)', checkpoint.name)
            if match:
                score = float(match.group(1))
                if score > best_score:
                    best_score = score
                    best_checkpoint = checkpoint
        
        if best_checkpoint:
            self.logger.info(f"Best checkpoint: {best_checkpoint} (score: {best_score:.4f})")
        
        return best_checkpoint

