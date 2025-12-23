"""
Arrow format utilities for zero-copy operations.
"""
import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
from pathlib import Path
from typing import Optional, Union, List, Dict, Any

from ..logging.logger import get_logger

logger = get_logger(__name__)


class ArrowUtils:
    """Utilities for Arrow format operations."""
    
    @staticmethod
    def numpy_to_arrow(array: np.ndarray, name: str = "data") -> pa.Array:
        """
        Convert NumPy array to Arrow array.
        
        Args:
            array: NumPy array
            name: Name for the array
            
        Returns:
            PyArrow Array
        """
        if array.dtype == np.float32:
            return pa.array(array, type=pa.float32())
        elif array.dtype == np.float64:
            return pa.array(array, type=pa.float64())
        elif array.dtype == np.int32:
            return pa.array(array, type=pa.int32())
        elif array.dtype == np.int64:
            return pa.array(array, type=pa.int64())
        else:
            return pa.array(array)
    
    @staticmethod
    def arrow_to_numpy(array: pa.Array) -> np.ndarray:
        """
        Convert Arrow array to NumPy array.
        
        Args:
            array: PyArrow Array
            
        Returns:
            NumPy array
        """
        return array.to_numpy()
    
    @staticmethod
    def save_array(
        array: np.ndarray,
        file_path: Union[str, Path],
        name: str = "data",
        compression: str = "snappy"
    ):
        """
        Save NumPy array to Arrow format.
        
        Args:
            array: NumPy array
            file_path: Output file path
            name: Name for the array
            compression: Compression type
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        arrow_array = ArrowUtils.numpy_to_arrow(array, name)
        schema = pa.schema([(name, arrow_array.type)])
        table = pa.table({name: arrow_array}, schema=schema)
        
        pq.write_table(table, file_path, compression=compression)
        logger.debug(f"Saved array to {file_path}")
    
    @staticmethod
    def load_array(
        file_path: Union[str, Path],
        name: str = "data"
    ) -> np.ndarray:
        """
        Load NumPy array from Arrow format.
        
        Args:
            file_path: Input file path
            name: Name of the array
            
        Returns:
            NumPy array
        """
        table = pq.read_table(file_path)
        array = table[name].to_numpy()
        logger.debug(f"Loaded array from {file_path}")
        return array
    
    @staticmethod
    def save_dict(
        data: Dict[str, Any],
        file_path: Union[str, Path],
        compression: str = "snappy"
    ):
        """
        Save dictionary to Arrow format.
        
        Args:
            data: Dictionary with array values
            file_path: Output file path
            compression: Compression type
        """
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        arrow_dict = {}
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                arrow_dict[key] = ArrowUtils.numpy_to_arrow(value, key)
            elif isinstance(value, (list, tuple)):
                arrow_dict[key] = pa.array(value)
            else:
                # Convert scalar to array
                arrow_dict[key] = pa.array([value])
        
        table = pa.table(arrow_dict)
        pq.write_table(table, file_path, compression=compression)
        logger.debug(f"Saved dictionary to {file_path}")
    
    @staticmethod
    def load_dict(file_path: Union[str, Path]) -> Dict[str, np.ndarray]:
        """
        Load dictionary from Arrow format.
        
        Args:
            file_path: Input file path
            
        Returns:
            Dictionary with NumPy array values
        """
        table = pq.read_table(file_path)
        data = {name: col.to_numpy() for name, col in zip(table.column_names, table.columns)}
        logger.debug(f"Loaded dictionary from {file_path}")
        return data
    
    @staticmethod
    def save_sparse_matrix(
        matrix,
        file_path: Union[str, Path],
        compression: str = "snappy"
    ):
        """
        Save sparse matrix to Arrow format.
        
        Args:
            matrix: Sparse matrix (scipy.sparse)
            file_path: Output file path
            compression: Compression type
        """
        from scipy.sparse import csr_matrix
        
        if not isinstance(matrix, csr_matrix):
            matrix = csr_matrix(matrix)
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Save as COO format (row, col, data)
        coo = matrix.tocoo()
        
        data = {
            'row': ArrowUtils.numpy_to_arrow(coo.row.astype(np.int64), 'row'),
            'col': ArrowUtils.numpy_to_arrow(coo.col.astype(np.int64), 'col'),
            'data': ArrowUtils.numpy_to_arrow(coo.data.astype(np.float32), 'data'),
            'shape': pa.array([matrix.shape[0], matrix.shape[1]], type=pa.int64())
        }
        
        table = pa.table(data)
        pq.write_table(table, file_path, compression=compression)
        logger.debug(f"Saved sparse matrix to {file_path}")
    
    @staticmethod
    def load_sparse_matrix(file_path: Union[str, Path]):
        """
        Load sparse matrix from Arrow format.
        
        Args:
            file_path: Input file path
            
        Returns:
            Sparse matrix (scipy.sparse.csr_matrix)
        """
        from scipy.sparse import csr_matrix
        
        table = pq.read_table(file_path)
        
        row = table['row'].to_numpy()
        col = table['col'].to_numpy()
        data = table['data'].to_numpy()
        shape = tuple(table['shape'].to_numpy())
        
        matrix = csr_matrix((data, (row, col)), shape=shape)
        logger.debug(f"Loaded sparse matrix from {file_path}")
        return matrix

