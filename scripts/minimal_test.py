#!/usr/bin/env python3
"""
Minimal test - imports one package at a time to find the crash.
"""
import sys
import os

print(f"Python: {sys.executable}")
print(f"Version: {sys.version}")
print(f"PID: {os.getpid()}")
sys.stdout.flush()

# Test basic imports one by one
tests = [
    ("sys", lambda: __import__("sys")),
    ("os", lambda: __import__("os")),
    ("numpy", lambda: __import__("numpy")),
    ("polars", lambda: __import__("polars")),
    ("sklearn", lambda: __import__("sklearn")),
    ("scipy", lambda: __import__("scipy")),
]

for name, import_func in tests:
    try:
        print(f"Testing {name}...", end=" ", flush=True)
        import_func()
        print("OK")
        sys.stdout.flush()
    except Exception as e:
        print(f"FAILED: {e}")
        sys.exit(1)

print("All basic imports OK")
sys.exit(0)

