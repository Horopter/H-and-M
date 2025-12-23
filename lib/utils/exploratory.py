"""
Exploratory Data Analysis utilities with comprehensive visualizations.
"""
import numpy as np
import polars as pl
from typing import Dict, Any, Optional
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

from ..config import get_config
from ..logging.logger import get_logger
from .visualization import Visualizer
from .arrow_storage import ArrowStorage

logger = get_logger(__name__)


class ExploratoryAnalyzer:
    """Perform exploratory data analysis."""
    
    def __init__(self, config=None, output_dir: Optional[str] = None):
        """Initialize exploratory analyzer."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir) if output_dir else Path(self.config.output_dir) / "exploratory"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer = Visualizer(config, str(self.output_dir))
        self.arrow_storage = ArrowStorage(config, str(Path(self.config.output_dir) / "arrow_data"))
    
    def analyze(self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """
        Perform comprehensive EDA.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame (optional)
            
        Returns:
            Dictionary with EDA results
        """
        self.logger.info("Performing exploratory data analysis")
        
        results = {}
        
        # Basic statistics
        results['train_stats'] = {
            'n_samples': len(train_df),
            'n_features': len(train_df.columns),
            'columns': train_df.columns
        }
        
        results['val_stats'] = {
            'n_samples': len(val_df),
            'n_features': len(val_df.columns)
        }
        
        if test_df is not None:
            results['test_stats'] = {
                'n_samples': len(test_df),
                'n_features': len(test_df.columns)
            }
        
        # Label distribution
        if self.config.label_column in train_df.columns:
            label_counts = train_df[self.config.label_column].value_counts().to_dict()
            results['label_distribution'] = label_counts
            self.logger.info(f"Label distribution: {label_counts}")
        
        # Text statistics
        if self.config.text_column in train_df.columns:
            train_texts = train_df[self.config.text_column].to_list()
            text_lengths = [len(str(t)) for t in train_texts]
            
            results['text_stats'] = {
                'mean_length': np.mean(text_lengths),
                'std_length': np.std(text_lengths),
                'min_length': np.min(text_lengths),
                'max_length': np.max(text_lengths),
                'median_length': np.median(text_lengths)
            }
        
        # Missing values
        results['missing_values'] = {
            'train': train_df.null_count().to_dict(),
            'val': val_df.null_count().to_dict()
        }
        
        # Generate visualizations
        self.logger.info("Generating EDA visualizations...")
        
        # Label distribution plot
        if self.config.label_column in train_df.columns:
            y_train = train_df[self.config.label_column].to_numpy()
            self.visualizer.plot_label_distribution(
                y_train,
                save_path=str(self.output_dir / "label_distribution.png")
            )
        
        # Text length distribution
        if self.config.text_column in train_df.columns:
            train_texts = train_df[self.config.text_column].to_list()
            text_lengths = [len(str(t)) for t in train_texts]
            self.visualizer.plot_text_length_distribution(
                text_lengths,
                save_path=str(self.output_dir / "text_length_distribution.png")
            )
        
        # Save data to Arrow format
        self.logger.info("Saving EDA data to Arrow format...")
        self.arrow_storage.save_dataframe(train_df, "train_data", "exploratory")
        self.arrow_storage.save_dataframe(val_df, "val_data", "exploratory")
        if test_df is not None:
            self.arrow_storage.save_dataframe(test_df, "test_data", "exploratory")
        
        # Save statistics
        stats_df = pl.DataFrame([results])
        self.arrow_storage.save_dataframe(stats_df, "eda_statistics", "exploratory")
        
        # Save label distribution
        if self.config.label_column in train_df.columns:
            label_df = train_df.select([self.config.label_column])
            self.arrow_storage.save_dataframe(label_df, "labels", "exploratory")
        
        self.logger.info("EDA complete")
        return results

