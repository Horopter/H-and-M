"""
Main entry point for the training pipeline.
"""
import argparse
from typing import Dict, Any, Optional

from .config import Config, get_config, set_config
from .logging.logger import get_logger
from .pipeline.stage_pipeline import StagePipeline
from .training.mlflow_tracker import MLFlowTracker
from .utils.duckdb_reporter import DuckDBReporter
from .utils.stats_analysis import StatisticalAnalyzer

logger = get_logger(__name__)


def train_pipeline(config_dict: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Main training pipeline function.
    
    Args:
        config_dict: Optional configuration dictionary
        
    Returns:
        Dictionary with training results
    """
    # Setup configuration
    if config_dict:
        config = Config(config_dict)
        set_config(config)
    else:
        config = get_config()
    
    config.ensure_directories()
    
    logger.info("Starting stage-based training pipeline")
    logger.info(f"Configuration: {config.to_dict()}")
    logger.info(f"Using consistent random seed: {config.random_state}")
    
    # Initialize stage pipeline
    pipeline = StagePipeline(config)
    
    # Initialize optional trackers
    mlflow_tracker = None
    duckdb_reporter = None
    
    try:
        mlflow_tracker = MLFlowTracker(config, experiment_name="main_experiment")
    except (ImportError, AttributeError, ValueError) as e:
        logger.debug(f"MLFlow tracker not available: {e}")
    except Exception as e:
        logger.warning(f"Could not initialize MLFlow tracker: {e}")
    
    try:
        duckdb_reporter = DuckDBReporter(config)
    except (ImportError, AttributeError, ValueError) as e:
        logger.debug(f"DuckDB reporter not available: {e}")
    except Exception as e:
        logger.warning(f"Could not initialize DuckDB reporter: {e}")
    
    # Run all stages
    results = pipeline.run_all_stages(
        experiment_name="main_experiment",
        mlflow_tracker=mlflow_tracker,
        duckdb_reporter=duckdb_reporter
    )
    
    logger.info("Stage-based training pipeline completed")
    
    return results


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description='GPU-First ML Training Pipeline')
    
    parser.add_argument(
        '--data-path',
        type=str,
        default='data/',
        help='Path to data directory'
    )
    parser.add_argument(
        '--checkpoint-dir',
        type=str,
        default='checkpoints/',
        help='Path to checkpoint directory'
    )
    parser.add_argument(
        '--log-dir',
        type=str,
        default='logs/',
        help='Path to log directory'
    )
    parser.add_argument(
        '--models',
        nargs='+',
        default=['logreg', 'svm', 'bayesian', 'xgboost', 'neural_net'],
        help='Models to train'
    )
    parser.add_argument(
        '--use-gpu',
        action='store_true',
        help='Use GPU if available'
    )
    parser.add_argument(
        '--subset-size',
        type=float,
        default=0.2,
        help='Fraction of data to use for CV/grid search (default: 0.2 = 20%%)'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds'
    )
    
    args = parser.parse_args()
    
    # Create configuration
    config_dict = {
        'data_path': args.data_path,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'models': args.models,
        'use_gpu': args.use_gpu,
        'subset_size': args.subset_size,
        'cv_folds': args.cv_folds
    }
    
    # Run pipeline
    results = train_pipeline(config_dict)
    
    print("\nTraining Results:")
    print("=" * 80)
    for model_type, result in results.get('results', {}).items():
        if 'error' in result:
            print(f"{model_type}: ERROR - {result['error']}")
        else:
            metrics = result.get('metrics', {})
            print(f"{model_type}:")
            for metric_name, metric_value in metrics.items():
                print(f"  {metric_name}: {metric_value:.4f}")
    
    return results


if __name__ == '__main__':
    main()

