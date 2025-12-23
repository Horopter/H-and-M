"""
Aggressive garbage collection utilities for memory management.
"""
import gc
from typing import Optional, Callable, Any
import numpy as np

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    import cudf
    CUDF_AVAILABLE = True
except ImportError:
    CUDF_AVAILABLE = False


class AggressiveGC:
    """Aggressive garbage collection manager."""
    
    def __init__(self, enabled: bool = True, verbose: bool = False):
        """
        Initialize aggressive GC manager.
        
        Args:
            enabled: Whether aggressive GC is enabled
            verbose: Whether to print GC statistics
        """
        self.enabled = enabled
        self.verbose = verbose
        self._gc_threshold = gc.get_threshold()
    
    def collect(self, generation: int = 2, aggressive: bool = True) -> tuple:
        """
        Perform hyper-aggressive garbage collection.
        
        Args:
            generation: GC generation to collect (0, 1, or 2)
            aggressive: If True, perform multiple passes (HYPER-AGGRESSIVE: 5 passes)
        
        Returns:
            Tuple of (collected, uncollectable) objects
        """
        if not self.enabled:
            return (0, 0)
        
        collected = 0
        uncollectable = 0
        
        if aggressive:
            # HYPER-AGGRESSIVE: 5 passes for maximum collection
            for _ in range(5):
                result = gc.collect(generation)
                collected += result[0] if isinstance(result, tuple) else result
                uncollectable += result[1] if isinstance(result, tuple) and len(result) > 1 else 0
        else:
            result = gc.collect(generation)
            collected = result[0] if isinstance(result, tuple) else result
            uncollectable = result[1] if isinstance(result, tuple) and len(result) > 1 else 0
        
        if self.verbose:
            print(f"GC: Collected {collected} objects, {uncollectable} uncollectable")
        
        return (collected, uncollectable)
    
    def collect_all(self) -> tuple:
        """
        Collect all generations hyper-aggressively.
        
        Returns:
            Tuple of (collected, uncollectable) objects
        """
        if not self.enabled:
            return (0, 0)
        
        total_collected = 0
        total_uncollectable = 0
        
        # HYPER-AGGRESSIVE: Collect all generations multiple times
        for _ in range(2):  # Two full passes
            for gen in range(3):  # All generations
                result = self.collect(gen, aggressive=True)
                total_collected += result[0]
                total_uncollectable += result[1]
        
        if self.verbose:
            print(f"GC All: Collected {total_collected} objects, {total_uncollectable} uncollectable")
        
        return (total_collected, total_uncollectable)
    
    def clear_caches(self):
        """Clear various caches hyper-aggressively to free memory."""
        if not self.enabled:
            return
        
        # Clear NumPy caches
        try:
            np.seterr(all='ignore')  # Suppress warnings during cache clearing
        except:
            pass
        
        # NOTE: sys.modules.clear() is DANGEROUS - removed to prevent breaking imports
        
        # HYPER-AGGRESSIVE: Clear PyTorch caches multiple times
        if TORCH_AVAILABLE:
            try:
                if torch.cuda.is_available():
                    # Multiple cache clears
                    for _ in range(3):
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                        torch.cuda.synchronize()
                # Clear PyTorch JIT cache
                torch.jit.clear_class_registry()
            except:
                pass
        
        # HYPER-AGGRESSIVE: Clear CuPy caches multiple times
        if CUPY_AVAILABLE:
            try:
                for _ in range(3):
                    mempool = cp.get_default_memory_pool()
                    mempool.free_all_blocks()
                    pinned_mempool = cp.get_default_pinned_memory_pool()
                    pinned_mempool.free_all_blocks()
            except:
                pass
        
        # Clear cuDF caches
        if CUDF_AVAILABLE:
            try:
                import cudf
                # cuDF doesn't have explicit cache clearing, but GC should handle it
                pass
            except:
                pass
    
    def force_collect(self) -> tuple:
        """
        Force aggressive garbage collection with cache clearing.
        
        Returns:
            Tuple of (collected, uncollectable) objects
        """
        if not self.enabled:
            return (0, 0)
        
        # Clear caches first
        self.clear_caches()
        
        # Then collect garbage
        result = self.collect_all()
        
        return result
    
    def get_memory_stats(self) -> dict:
        """
        Get current memory statistics.
        
        Returns:
            Dictionary with memory statistics
        """
        stats = {
            'gc_counts': gc.get_count(),
            'gc_threshold': gc.get_threshold(),
            'gc_stats': gc.get_stats(),
        }
        
        # Add GPU memory if available
        if TORCH_AVAILABLE and torch.cuda.is_available():
            try:
                stats['gpu_allocated'] = torch.cuda.memory_allocated() / 1024**3  # GB
                stats['gpu_reserved'] = torch.cuda.memory_reserved() / 1024**3  # GB
                stats['gpu_max_allocated'] = torch.cuda.max_memory_allocated() / 1024**3  # GB
            except:
                pass
        
        if CUPY_AVAILABLE:
            try:
                mempool = cp.get_default_memory_pool()
                stats['cupy_used'] = mempool.used_bytes() / 1024**3  # GB
                stats['cupy_total'] = mempool.total_bytes() / 1024**3  # GB
            except:
                pass
        
        return stats
    
    def optimize_thresholds(self, gen0: int = 100, gen1: int = 5, gen2: int = 5):
        """
        Optimize GC thresholds for HYPER-AGGRESSIVE frequent collection.
        
        Args:
            gen0: Generation 0 threshold (lower = more frequent)
            gen1: Generation 1 threshold (lower = more frequent)
            gen2: Generation 2 threshold (lower = more frequent)
        """
        if not self.enabled:
            return
        
        gc.set_threshold(gen0, gen1, gen2)
        if self.verbose:
            print(f"GC thresholds set to: {gen0}, {gen1}, {gen2}")
    
    def reset_thresholds(self):
        """Reset GC thresholds to original values."""
        if not self.enabled:
            return
        
        gc.set_threshold(*self._gc_threshold)
        if self.verbose:
            print(f"GC thresholds reset to: {self._gc_threshold}")


# Global instance
_gc_manager: Optional[AggressiveGC] = None


def get_gc_manager(enabled: bool = True, verbose: bool = False) -> AggressiveGC:
    """
    Get or create global GC manager.
    
    Args:
        enabled: Whether GC is enabled
        verbose: Whether to print statistics
    
    Returns:
        AggressiveGC instance
    """
    global _gc_manager
    if _gc_manager is None:
        _gc_manager = AggressiveGC(enabled=enabled, verbose=verbose)
    return _gc_manager


def collect_after_chunk(chunk_id: Optional[int] = None, aggressive: bool = True):
    """
    Convenience function to collect after processing a chunk (HYPER-AGGRESSIVE).
    
    Args:
        chunk_id: Optional chunk ID for logging
        aggressive: Whether to use hyper-aggressive collection (always True)
    """
    gc_manager = get_gc_manager()
    # Always use hyper-aggressive collection
    gc_manager.force_collect()


def collect_after_operation(operation_name: Optional[str] = None, aggressive: bool = True):
    """
    Convenience function to collect after a large operation (HYPER-AGGRESSIVE).
    
    Args:
        operation_name: Optional operation name for logging
        aggressive: Whether to use hyper-aggressive collection (always True)
    """
    gc_manager = get_gc_manager()
    # Always use hyper-aggressive collection with cache clearing
    gc_manager.force_collect()


def context_manager_collect(func: Callable) -> Callable:
    """
    Decorator to automatically collect garbage after function execution.
    
    Args:
        func: Function to decorate
    
    Returns:
        Decorated function
    """
    def wrapper(*args, **kwargs):
        try:
            result = func(*args, **kwargs)
            return result
        finally:
            collect_after_operation(func.__name__, aggressive=True)
    
    return wrapper


def memory_efficient_chunked_processing(
    items: list,
    process_func: Callable,
    chunk_size: int = 1000,
    collect_after_each: bool = True,
    aggressive_gc: bool = True
) -> list:
    """
    Process items in chunks with automatic garbage collection.
    
    Args:
        items: List of items to process
        process_func: Function to process each chunk
        chunk_size: Size of each chunk
        collect_after_each: Whether to collect after each chunk
        aggressive_gc: Whether to use aggressive GC
    
    Returns:
        List of processed results
    """
    results = []
    gc_manager = get_gc_manager()
    
    for i in range(0, len(items), chunk_size):
        chunk = items[i:i+chunk_size]
        
        # Process chunk
        chunk_result = process_func(chunk)
        results.append(chunk_result)
        
        # Clear chunk reference
        del chunk
        
        # Collect garbage after each chunk
        if collect_after_each:
            if aggressive_gc:
                gc_manager.collect_all()
            else:
                gc_manager.collect()
    
    return results

