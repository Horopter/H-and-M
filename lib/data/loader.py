"""
Polars-based data loading with Arrow I/O.
"""
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import numpy as np

from ..config import get_config
from ..logging.logger import get_logger
from ..utils.gc_utils import collect_after_chunk

logger = get_logger(__name__)


class DataLoader:
    """Polars-based data loader with Arrow format support."""
    
    def __init__(self, config=None):
        """
        Initialize data loader.
        
        Args:
            config: Configuration object (uses global config if None)
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def load_csv(
        self,
        file_path: str,
        text_column: Optional[str] = None,
        label_column: Optional[str] = None,
        id_column: Optional[str] = None,
        chunked: bool = True
    ) -> pl.DataFrame:
        """
        Load CSV file using Polars with chunked processing.
        
        Args:
            file_path: Path to CSV file
            text_column: Name of text column (uses config default if None)
            label_column: Name of label column (uses config default if None)
            id_column: Name of ID column (uses config default if None)
            chunked: If True, load in chunks to save memory
            
        Returns:
            Polars DataFrame
        """
        file_path = Path(self.config.data_path) / file_path
        self.logger.info(f"Loading CSV from {file_path}")
        
        try:
            text_col = text_column or self.config.text_column
            label_col = label_column or self.config.label_column
            id_col = id_column or self.config.id_column
            
            if chunked and self.config.chunk_size > 0:
                # Use scan_csv for lazy evaluation and chunked processing
                try:
                    # Try lazy scan approach (more memory efficient)
                    lazy_df = pl.scan_csv(
                        file_path,
                        try_parse_dates=False,
                        encoding='utf8-lossy'
                    )
                    
                    # Collect in chunks
                    chunk_size = self.config.chunk_size
                    chunks = []
                    total_rows = 0
                    
                    # Get total row count for progress (optional)
                    try:
                        count_result = lazy_df.select(pl.count()).collect()
                        if len(count_result) > 0:
                            row_count = count_result[0, 0] if hasattr(count_result, '__getitem__') else None
                        else:
                            row_count = None
                    except:
                        row_count = None
                    
                    offset = 0
                    while True:
                        chunk_df = lazy_df.slice(offset, chunk_size).collect()
                        
                        # Ensure chunk_df is a DataFrame
                        if not isinstance(chunk_df, pl.DataFrame):
                            raise TypeError(f"Expected DataFrame, got {type(chunk_df)}")
                        
                        if len(chunk_df) == 0:
                            break
                        
                        # Validate columns on first chunk
                        if offset == 0:
                            if text_col not in chunk_df.columns:
                                raise ValueError(f"Text column '{text_col}' not found in {file_path}")
                        
                        # Process chunk
                        chunk_df = chunk_df.with_columns([
                            pl.col(text_col).cast(pl.Utf8).alias(text_col)
                        ])
                        
                        if label_col and label_col in chunk_df.columns:
                            chunk_df = chunk_df.with_columns([
                                pl.col(label_col).cast(pl.Int64).alias(label_col)
                            ])
                        
                        if id_col and id_col in chunk_df.columns:
                            chunk_df = chunk_df.with_columns([
                                pl.col(id_col).cast(pl.Int64).alias(id_col)
                            ])
                        
                        chunks.append(chunk_df)
                        chunk_len = len(chunk_df)
                        total_rows += chunk_len
                        offset += chunk_size
                        
                        # Aggressive GC after each chunk
                        collect_after_chunk(offset // chunk_size, aggressive=True)
                        
                        # Clear chunk reference
                        del chunk_df
                        
                        if chunk_len < chunk_size:
                            break
                    
                    df = pl.concat(chunks)
                    
                    # Final validation
                    if not isinstance(df, pl.DataFrame):
                        raise TypeError(f"Expected DataFrame after concat, got {type(df)}")
                    
                    self.logger.info(f"Loaded {len(df)} rows in {len(chunks)} chunks, {len(df.columns)} columns")
                except Exception as scan_error:
                    # Fallback to regular read if scan_csv fails
                    self.logger.debug(f"scan_csv failed, using regular read: {scan_error}")
                    df = pl.read_csv(
                        file_path,
                        try_parse_dates=False,
                        encoding='utf8-lossy'
                    )
                    
                    if text_col not in df.columns:
                        raise ValueError(f"Text column '{text_col}' not found in {file_path}")
                    
                    df = df.with_columns([
                        pl.col(text_col).cast(pl.Utf8).alias(text_col)
                    ])
                    
                    if label_col and label_col in df.columns:
                        df = df.with_columns([
                            pl.col(label_col).cast(pl.Int64).alias(label_col)
                        ])
                    
                    if id_col and id_col in df.columns:
                        df = df.with_columns([
                            pl.col(id_col).cast(pl.Int64).alias(id_col)
                        ])
                    
                    self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            else:
                df = pl.read_csv(
                    file_path,
                    try_parse_dates=False,
                    encoding='utf8-lossy'
                )
                
                # Ensure df is a DataFrame
                if not isinstance(df, pl.DataFrame):
                    raise TypeError(f"Expected DataFrame, got {type(df)}")
                
                if text_col not in df.columns:
                    raise ValueError(f"Text column '{text_col}' not found in {file_path}")
                
                df = df.with_columns([
                    pl.col(text_col).cast(pl.Utf8).alias(text_col)
                ])
                
                if label_col and label_col in df.columns:
                    df = df.with_columns([
                        pl.col(label_col).cast(pl.Int64).alias(label_col)
                    ])
                
                if id_col and id_col in df.columns:
                    df = df.with_columns([
                        pl.col(id_col).cast(pl.Int64).alias(id_col)
                    ])
                
                self.logger.info(f"Loaded {len(df)} rows, {len(df.columns)} columns")
            
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading CSV from {file_path}: {e}")
            raise
    
    def remove_duplicates_from_val(
        self,
        val_df: pl.DataFrame,
        test_df: Optional[pl.DataFrame] = None,
        text_column: Optional[str] = None
    ) -> pl.DataFrame:
        """
        Remove duplicate texts from validation set that appear in test set.
        
        Args:
            val_df: Validation DataFrame
            test_df: Test DataFrame (optional)
            text_column: Name of text column
            
        Returns:
            Validation DataFrame with duplicates removed
        """
        if test_df is None or len(test_df) == 0:
            return val_df
        
        if len(val_df) == 0:
            self.logger.warning("Validation DataFrame is empty, nothing to filter")
            return val_df
        
        text_col = text_column or self.config.text_column
        
        # Get test texts as set for fast lookup (stripped)
        test_texts = set(str(t).strip() for t in test_df[text_col].to_list())
        
        # Filter validation to exclude texts that appear in test
        # Convert to Python list, strip, and filter (more compatible across Polars versions)
        val_texts_list = val_df[text_col].cast(pl.Utf8).to_list()
        val_texts_stripped = [str(t).strip() if t is not None else "" for t in val_texts_list]
        
        # Create mask for rows NOT in test set
        mask = pl.Series([stripped not in test_texts for stripped in val_texts_stripped])
        val_df_clean = val_df.filter(mask)
        
        original_len = len(val_df)
        removed = original_len - len(val_df_clean)
        
        if removed > 0:
            self.logger.warning(
                f"Removed {removed} duplicate texts from validation set "
                f"that appear in test set ({original_len} -> {len(val_df_clean)})"
            )
        else:
            self.logger.debug("No duplicates found between validation and test sets")
        
        return val_df_clean
    
    def load_train_val_test(
        self,
        remove_val_test_duplicates: bool = True
    ) -> Tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
        """
        Load train, validation, and test datasets.
        
        Args:
            remove_val_test_duplicates: If True, remove duplicates from validation set
        
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        train_df = self.load_csv(
            self.config.train_file,
            self.config.text_column,
            self.config.label_column,
            self.config.id_column
        )
        
        val_df = self.load_csv(
            self.config.val_file,
            self.config.text_column,
            self.config.label_column,
            self.config.id_column
        )
        
        test_df = self.load_csv(
            self.config.test_file,
            self.config.text_column,
            None,  # No label in test
            self.config.id_column
        )
        
        # Remove duplicates from validation set that appear in test set
        if remove_val_test_duplicates:
            val_df = self.remove_duplicates_from_val(val_df, test_df)
        
        self.logger.info(
            f"Loaded datasets - Train: {len(train_df)}, "
            f"Val: {len(val_df)}, Test: {len(test_df)}"
        )
        
        return train_df, val_df, test_df
    
    def save_arrow(
        self,
        df: pl.DataFrame,
        file_path: str,
        compression: Optional[str] = None
    ):
        """
        Save DataFrame to Arrow/Parquet format.
        
        Args:
            df: Polars DataFrame
            file_path: Output file path
            compression: Compression type (default from config)
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        compression = compression or self.config.compression
        
        self.logger.info(f"Saving DataFrame to Arrow format: {file_path}")
        
        try:
            # Convert to Arrow table
            arrow_table = df.to_arrow()
            
            # Write as Parquet (Arrow format)
            pq.write_table(
                arrow_table,
                file_path,
                compression=compression,
                use_dictionary=True
            )
            
            self.logger.info(f"Saved {len(df)} rows to {file_path}")
            
        except Exception as e:
            self.logger.error(f"Error saving Arrow file {file_path}: {e}")
            raise
    
    def load_arrow(self, file_path: str) -> pl.DataFrame:
        """
        Load DataFrame from Arrow/Parquet format.
        
        Args:
            file_path: Input file path
            
        Returns:
            Polars DataFrame
        """
        file_path = Path(file_path)
        self.logger.info(f"Loading Arrow file: {file_path}")
        
        try:
            # Read Parquet
            arrow_table = pq.read_table(file_path)
            
            # Convert to Polars
            df = pl.from_arrow(arrow_table)
            
            self.logger.info(f"Loaded {len(df)} rows from {file_path}")
            return df
            
        except Exception as e:
            self.logger.error(f"Error loading Arrow file {file_path}: {e}")
            raise
    
    def to_arrow_table(self, df: pl.DataFrame) -> pa.Table:
        """
        Convert Polars DataFrame to Arrow Table.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            PyArrow Table
        """
        return df.to_arrow()
    
    def from_arrow_table(self, table: pa.Table) -> pl.DataFrame:
        """
        Convert Arrow Table to Polars DataFrame.
        
        Args:
            table: PyArrow Table
            
        Returns:
            Polars DataFrame
        """
        return pl.from_arrow(table)
    
    def validate_schema(
        self,
        df: pl.DataFrame,
        required_columns: List[str],
        dataset_name: str = "dataset"
    ) -> bool:
        """
        Validate DataFrame schema.
        
        Args:
            df: Polars DataFrame
            required_columns: List of required column names
            dataset_name: Name of dataset for logging
            
        Returns:
            True if valid, raises ValueError if not
        """
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            error_msg = f"{dataset_name} missing required columns: {missing}"
            self.logger.error(error_msg)
            raise ValueError(error_msg)
        
        self.logger.debug(f"{dataset_name} schema validation passed")
        return True
    
    def get_text_column(self, df: pl.DataFrame) -> pl.Series:
        """Get text column from DataFrame."""
        return df[self.config.text_column]
    
    def get_label_column(self, df: pl.DataFrame) -> Optional[pl.Series]:
        """Get label column from DataFrame if present."""
        if self.config.label_column in df.columns:
            return df[self.config.label_column]
        return None
    
    def get_id_column(self, df: pl.DataFrame) -> Optional[pl.Series]:
        """Get ID column from DataFrame if present."""
        if self.config.id_column in df.columns:
            return df[self.config.id_column]
        return None
    
    def check_missing_values(self, df: pl.DataFrame) -> Dict[str, int]:
        """
        Check for missing values in DataFrame.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            Dictionary of column name to missing count
        """
        missing = {}
        for col in df.columns:
            null_count = df[col].null_count()
            if null_count > 0:
                missing[col] = null_count
        
        if missing:
            self.logger.warning(f"Found missing values: {missing}")
        else:
            self.logger.debug("No missing values found")
        
        return missing
    
    def get_data_info(self, df: pl.DataFrame) -> Dict[str, Any]:
        """
        Get information about the dataset.
        
        Args:
            df: Polars DataFrame
            
        Returns:
            Dictionary with dataset information
        """
        info = {
            'rows': len(df),
            'columns': len(df.columns),
            'column_names': df.columns,
            'dtypes': {col: str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            'memory_usage': df.estimated_size()
        }
        
        # Add label distribution if label column exists
        if self.config.label_column in df.columns:
            label_counts = df[self.config.label_column].value_counts().sort('label')
            info['label_distribution'] = label_counts.to_dict()
        
        return info

