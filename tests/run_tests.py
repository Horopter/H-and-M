#!/usr/bin/env python3
"""
Comprehensive test runner for GPU-First ML Library.
"""
import unittest
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def run_all_tests():
    """Run all tests."""
    # Discover and run all tests
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # Add test modules
    test_modules = [
        'tests.test_constants',
        'tests.test_message_loader',
        'tests.test_chunked_processing',
        'tests.test_config_constants_integration',
        'tests.test_error_handling',
        'tests.test_data_loader',
        'tests.test_cv_stratified',
        'tests.test_gpu_fallback',
        'tests.test_models',
        'tests.test_validation',
        'tests.test_integration'
    ]
    
    for module_name in test_modules:
        try:
            module = __import__(module_name, fromlist=[''])
            tests = loader.loadTestsFromModule(module)
            suite.addTests(tests)
            print(f"✓ Loaded tests from {module_name}")
        except Exception as e:
            print(f"✗ Failed to load {module_name}: {e}")
    
    # Run tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    print(f"Success: {result.testsRun - len(result.failures) - len(result.errors)}")
    
    if result.failures:
        print("\nFailures:")
        for test, traceback in result.failures:
            print(f"  - {test}: {traceback[:200]}")
    
    if result.errors:
        print("\nErrors:")
        for test, traceback in result.errors:
            print(f"  - {test}: {traceback[:200]}")
    
    print("=" * 80)
    
    return 0 if result.wasSuccessful() else 1


if __name__ == '__main__':
    sys.exit(run_all_tests())

