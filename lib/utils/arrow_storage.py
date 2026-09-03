"""
Arrow format data storage utilities.
Save all intermediate data in Arrow format for future reference.
"""
import numpy as np
import polars as pl
import pyarrow as pa
import pyarrow.parquet as pq
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
from scipy.sparse import csr_matrix, issparse

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class ArrowStorage:
    """Store and retrieve data in Arrow/Parquet format."""
    
    def __init__(self, config=None, storage_dir: Optional[str] = None):
        """Initialize Arrow storage."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.storage_dir = Path(storage_dir) if storage_dir else Path(self.config.output_dir) / "arrow_data"
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    def save_dataframe(self, df: pl.DataFrame, name: str, stage: str = "general"):
        """Save Polars DataFrame as Parquet (Arrow format)."""
        save_path = self.storage_dir / stage / f"{name}.parquet"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(str(save_path))
        self.logger.debug(f"Saved DataFrame {name} to {save_path}")
        return save_path
    
    def save_numpy_array(self, array: np.ndarray, name: str, stage: str = "general", 
                        metadata: Optional[Dict[str, Any]] = None):
        """Save NumPy array as Parquet."""
        # Convert to Polars DataFrame
        if array.ndim == 1:
            df = pl.DataFrame({name: array})
        else:
            # Multi-dimensional array - flatten or save as multiple columns
            columns = {f"{name}_{i}": array[:, i] for i in range(array.shape[1])}
            df = pl.DataFrame(columns)
        
        # Add metadata as additional columns if provided
        if metadata:
            for key, value in metadata.items():
                if isinstance(value, (int, float, str, bool)):
                    df = df.with_columns([pl.lit(value).alias(f"_meta_{key}")])
        
        save_path = self.storage_dir / stage / f"{name}.parquet"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(str(save_path))
        self.logger.debug(f"Saved array {name} to {save_path}")
        return save_path
    
    def save_sparse_matrix(self, matrix: Union[csr_matrix, np.ndarray], name: str, stage: str = "general"):
        """Save sparse matrix as Parquet."""
        if matrix is None:
            raise ValueError(f"Cannot save empty matrix for {name}")
        if not issparse(matrix):
            try:
                matrix = csr_matrix(matrix)
                self.logger.info("Converted dense matrix to CSR for %s", name)
            except Exception as e:
                raise TypeError(f"Expected sparse matrix for {name}, got {type(matrix)}") from e
        
        # Convert to COO format for easier storage
        coo = matrix.tocoo()
        
        df = pl.DataFrame({
            'row': coo.row,
            'col': coo.col,
            'data': coo.data
        })
        
        # Save shape as metadata
        metadata = {
            'shape_0': matrix.shape[0],
            'shape_1': matrix.shape[1],
            'nnz': matrix.nnz
        }
        
        save_path = self.storage_dir / stage / f"{name}.parquet"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(str(save_path))
        
        # Save metadata separately
        metadata_path = self.storage_dir / stage / f"{name}_metadata.parquet"
        pl.DataFrame([metadata]).write_parquet(str(metadata_path))
        
        self.logger.debug(f"Saved sparse matrix {name} to {save_path}")
        return save_path
    
    def save_metrics(self, metrics: Dict[str, Any], name: str, stage: str = "general"):
        """Save metrics dictionary as Parquet."""
        # Flatten nested dictionaries
        flat_metrics = self._flatten_dict(metrics)
        
        df = pl.DataFrame([flat_metrics])
        save_path = self.storage_dir / stage / f"{name}_metrics.parquet"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.write_parquet(str(save_path))
        self.logger.debug(f"Saved metrics {name} to {save_path}")
        return save_path
    
    def save_cv_results(self, cv_results: Dict[str, Any], name: str, stage: str = "general"):
        """Save CV results as Parquet."""
        # Convert CV results to tabular format
        rows = []
        
        for model_name, results in cv_results.items():
            if 'error' not in results and 'metrics' in results:
                for metric_name, metric_data in results['metrics'].items():
                    row = {
                        'model': model_name,
                        'metric': metric_name,
                        'mean': metric_data.get('mean', 0),
                        'std': metric_data.get('std', 0),
                        'min': metric_data.get('min', 0),
                        'max': metric_data.get('max', 0)
                    }
                    if 'values' in metric_data:
                        row['n_folds'] = len(metric_data['values'])
                    rows.append(row)
        
        if rows:
            df = pl.DataFrame(rows)
            save_path = self.storage_dir / stage / f"{name}_cv_results.parquet"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            df.write_parquet(str(save_path))
            self.logger.debug(f"Saved CV results {name} to {save_path}")
            return save_path
        
        return None
    
    def save_embeddings(self, embeddings: Dict[str, np.ndarray], name: str, stage: str = "general"):
        """Save embeddings dictionary as Parquet files."""
        saved_paths = {}
        for emb_type, emb_array in embeddings.items():
            save_path = self.save_numpy_array(emb_array, f"{name}_{emb_type}", stage)
            saved_paths[emb_type] = save_path
            self.logger.info(f"Saved {emb_type} embeddings to {save_path}")
        return saved_paths
    
    def load_embeddings(self, name: str, stage: str = "general", emb_types: Optional[List[str]] = None) -> Optional[Dict[str, np.ndarray]]:
        """Load embeddings from Parquet files."""
        embeddings = {}
        stage_dir = self.storage_dir / stage
        
        if not stage_dir.exists():
            return None
        
        # If emb_types not specified, try to find all matching files
        if emb_types is None:
            pattern = f"{name}_*.parquet"
            files = list(stage_dir.glob(pattern))
            emb_types = [f.stem.replace(f"{name}_", "") for f in files]
        
        for emb_type in emb_types:
            emb_path = stage_dir / f"{name}_{emb_type}.parquet"
            if emb_path.exists():
                try:
                    df = pl.read_parquet(str(emb_path))
                    # Filter out metadata columns
                    data_columns = [c for c in df.columns if not c.startswith("_meta_")]
                    if len(data_columns) == 1:
                        array = df[data_columns[0]].to_numpy()
                    else:
                        # Multi-column array - stack columns
                        array = df.select(data_columns).to_numpy()
                    embeddings[emb_type] = array
                    self.logger.info(f"Loaded {emb_type} embeddings from {emb_path} (shape: {array.shape})")
                except Exception as e:
                    self.logger.warning(f"Could not load {emb_type} embeddings from {emb_path}: {e}")
        
        return embeddings if embeddings else None
    
    def embeddings_exist(self, name: str, stage: str = "general", emb_types: Optional[List[str]] = None) -> bool:
        """Check if embeddings exist."""
        stage_dir = self.storage_dir / stage
        if not stage_dir.exists():
            return False
        
        if emb_types is None:
            # Check for any embedding files with this name
            pattern = f"{name}_*.parquet"
            return len(list(stage_dir.glob(pattern))) > 0
        
        # Check for specific embedding types
        for emb_type in emb_types:
            emb_path = stage_dir / f"{name}_{emb_type}.parquet"
            if not emb_path.exists():
                return False
        return True
    
    def _flatten_dict(self, d: Dict[str, Any], parent_key: str = '', sep: str = '_') -> Dict[str, Any]:
        """Flatten nested dictionary."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(self._flatten_dict(v, new_key, sep=sep).items())
            elif isinstance(v, list):
                # Convert list to string representation or take first few elements
                if len(v) > 0 and isinstance(v[0], (int, float)):
                    items.append((new_key, v[0]))  # Take first element
                else:
                    items.append((new_key, str(v)))
            else:
                items.append((new_key, v))
        return dict(items)
    
    def load_dataframe(self, name: str, stage: str = "general") -> Optional[pl.DataFrame]:
        """Load DataFrame from Parquet."""
        load_path = self.storage_dir / stage / f"{name}.parquet"
        if load_path.exists():
            return pl.read_parquet(str(load_path))
        return None
    
    def load_sparse_matrix(self, name: str, stage: str = "general") -> Optional[csr_matrix]:
        """Load sparse matrix from Parquet."""
        load_path = self.storage_dir / stage / f"{name}.parquet"
        metadata_path = self.storage_dir / stage / f"{name}_metadata.parquet"
        
        if load_path.exists() and metadata_path.exists():
            df = pl.read_parquet(str(load_path))
            metadata_df = pl.read_parquet(str(metadata_path))
            
            if 'shape_0' not in metadata_df.columns or 'shape_1' not in metadata_df.columns:
                return None
            if len(metadata_df) == 0:
                return None
            shape_0 = metadata_df['shape_0'][0]
            shape_1 = metadata_df['shape_1'][0]
            
            # Reconstruct sparse matrix
            row = df['row'].to_numpy()
            col = df['col'].to_numpy()
            data = df['data'].to_numpy()
            
            return csr_matrix((data, (row, col)), shape=(shape_0, shape_1))
        
        return None
