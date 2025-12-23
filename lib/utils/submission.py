"""
Submission file generator - ONLY CSV file generated in the entire pipeline.
"""
import numpy as np
import polars as pl
from typing import Optional, Dict, Any
from pathlib import Path

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class SubmissionGenerator:
    """Generate submission CSV file (ONLY CSV file in pipeline)."""
    
    def __init__(self, config=None):
        """Initialize submission generator."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def generate_submission(
        self,
        predictions: np.ndarray,
        test_ids: Optional[np.ndarray] = None,
        output_path: Optional[str] = None
    ) -> Path:
        """
        Generate submission CSV file.
        
        Args:
            predictions: Model predictions (probabilities or class labels)
            test_ids: Test sample IDs (optional)
            output_path: Path to save submission file
            
        Returns:
            Path to saved submission file
        """
        self.logger.info("Generating submission file (ONLY CSV in pipeline)")
        
        # Determine output path
        if output_path is None:
            submissions_dir = Path(self.config.output_dir) / "submissions"
            submissions_dir.mkdir(parents=True, exist_ok=True)
            output_path = submissions_dir / "submission.csv"
        else:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Create submission DataFrame
        if test_ids is not None:
            submission_df = pl.DataFrame({
                'id': test_ids,
                'prediction': predictions
            })
        else:
            # Use index as ID
            submission_df = pl.DataFrame({
                'id': np.arange(len(predictions)),
                'prediction': predictions
            })
        
        # Write CSV (ONLY CSV file in entire pipeline)
        submission_df.write_csv(str(output_path))
        
        self.logger.info(f"Submission file saved to {output_path}")
        self.logger.info(f"Submission shape: {submission_df.shape}")
        self.logger.info(f"Predictions range: [{np.min(predictions):.4f}, {np.max(predictions):.4f}]")
        
        return output_path

