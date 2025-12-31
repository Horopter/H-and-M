"""HVSM runtime hooks for RAPIDS memory configuration.

This module is auto-imported by Python if it is on sys.path.
Enable it by setting HVSM_ENABLE_RMM=1 in the environment.
"""

from __future__ import annotations

import os


def _truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _reset_cupy_allocator() -> None:
    import cupy as cp

    cp.cuda.set_allocator(None)
    try:
        cp.cuda.set_pinned_memory_allocator(None)
    except Exception:
        pass


def _init_rmm() -> None:
    if not _truthy(os.environ.get("HVSM_ENABLE_RMM")):
        return
    try:
        import rmm
        import cupy as cp

        managed = _truthy(os.environ.get("RMM_MANAGED_MEMORY", "1"))
        pool = _truthy(os.environ.get("RMM_POOL", "1"))
        pool_size = os.environ.get("RMM_POOL_SIZE")
        initial_pool_size = int(pool_size) if pool_size else None

        rmm.reinitialize(
            pool_allocator=pool,
            managed_memory=managed,
            initial_pool_size=initial_pool_size,
        )
        if _truthy(os.environ.get("HVSM_RMM_CUPY")):
            from rmm.allocators.cupy import rmm_cupy_allocator

            cp.cuda.set_allocator(rmm_cupy_allocator)
        else:
            # Force Cupy back to the default allocator to avoid RMM/Cupy conflicts.
            _reset_cupy_allocator()

        try:
            import cudf

            cudf.set_option("spill", True)
            if not _truthy(os.environ.get("HVSM_RMM_CUPY")):
                _reset_cupy_allocator()
        except Exception:
            pass
    except Exception as exc:
        if _truthy(os.environ.get("HVSM_RMM_DEBUG")):
            print(f"[sitecustomize] RMM init skipped: {exc}", flush=True)


_init_rmm()

# Ensure any later cudf/cuml import doesn't flip Cupy back to the RMM allocator.
if _truthy(os.environ.get("HVSM_ENABLE_RMM")) and not _truthy(os.environ.get("HVSM_RMM_CUPY")):
    import builtins

    _orig_import = builtins.__import__

    def _patched_import(name, globals=None, locals=None, fromlist=(), level=0):
        mod = _orig_import(name, globals, locals, fromlist, level)
        root = name.split(".", 1)[0]
        if root in {"cudf", "cuml"}:
            try:
                _reset_cupy_allocator()
            except Exception:
                pass
        return mod

    builtins.__import__ = _patched_import
