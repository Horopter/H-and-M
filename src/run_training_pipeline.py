#!/usr/bin/env python3
"""
Main execution script for GPU-first ML training pipeline.
Integrates MLFlow, DuckDB, and statistical analyses.
"""
import argparse
import sys
from pathlib import Path
from datetime import datetime
import json

# Add lib to path (go up from src/ to root)
sys.path.insert(0, str(Path(__file__).parent.parent))

from lib.config import Config, get_config, set_config
from lib.main import train_pipeline
from lib.training.mlflow_tracker import MLFlowTracker
from lib.utils.duckdb_reporter import DuckDBReporter
from lib.utils.stats_analysis import StatisticalAnalyzer
from lib.logging.logger import get_logger

logger = get_logger(__name__)


def main():
    """Main execution function."""
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
        '--output-dir',
        type=str,
        default='outputs/',
        help='Path to output directory'
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
        default=0.1,
        help='Fraction of data to use for CV/grid search'
    )
    parser.add_argument(
        '--cv-folds',
        type=int,
        default=5,
        help='Number of CV folds (must be stratified)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='MLFlow experiment name'
    )
    parser.add_argument(
        '--mlflow-tracking-uri',
        type=str,
        default=None,
        help='MLFlow tracking URI'
    )
    parser.add_argument(
        '--duckdb-path',
        type=str,
        default=None,
        help='Path to DuckDB database (default: {output_dir}/results.duckdb)'
    )
    parser.add_argument(
        '--run-stats',
        action='store_true',
        help='Run statistical analyses (ANOVA, Tukey HSD, UMAP)'
    )
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup configuration
    config_dict = {
        'data_path': args.data_path,
        'checkpoint_dir': args.checkpoint_dir,
        'log_dir': args.log_dir,
        'output_dir': args.output_dir,
        'models': args.models,
        'use_gpu': args.use_gpu,
        'subset_size': args.subset_size,
        'cv_folds': args.cv_folds
    }
    
    config = Config(config_dict)
    set_config(config)
    config.ensure_directories()
    
    # Generate experiment ID
    experiment_id = f"exp_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    experiment_name = args.experiment_name or f"kaggle_ml_{experiment_id}"
    
    logger.info("=" * 80)
    logger.info("GPU-FIRST ML TRAINING PIPELINE")
    logger.info("=" * 80)
    logger.info(f"Experiment ID: {experiment_id}")
    logger.info(f"Experiment Name: {experiment_name}")
    logger.info(f"Models: {args.models}")
    logger.info(f"CV Folds: {args.cv_folds} (stratified)")
    logger.info(f"Subset Size: {args.subset_size}")
    logger.info(f"Use GPU: {args.use_gpu}")
    logger.info("=" * 80)
    
    # Initialize MLFlow
    mlflow_tracker = None
    try:
        if args.mlflow_tracking_uri:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow_tracker = MLFlowTracker(config, experiment_name)
        mlflow_tracker.start_run(run_name=experiment_id)
        mlflow_tracker.log_params(config.to_dict())
    except Exception as e:
        logger.warning(f"MLFlow initialization failed: {e}")
    
    # Initialize DuckDB
    duckdb_reporter = None
    try:
        duckdb_path = args.duckdb_path or str(Path(args.output_dir) / "results.duckdb")
        duckdb_reporter = DuckDBReporter(config, duckdb_path)
        duckdb_reporter.log_experiment(experiment_id, experiment_name, config.to_dict())
    except Exception as e:
        logger.warning(f"DuckDB initialization failed: {e}")
    
    # Initialize statistical analyzer
    stats_analyzer = None
    if args.run_stats:
        try:
            stats_analyzer = StatisticalAnalyzer(config)
        except Exception as e:
            logger.warning(f"Statistical analyzer initialization failed: {e}")
    
    try:
        # Run training pipeline
        logger.info("Starting training pipeline...")
        results = train_pipeline(config_dict)
        
        # Log results to MLFlow and DuckDB
        all_cv_results = {}
        
        for model_type, result in results.get('results', {}).items():
            if 'error' in result:
                logger.error(f"{model_type}: {result['error']}")
                continue
            
            metrics = result.get('metrics', {})
            
            # Log to MLFlow
            if mlflow_tracker:
                try:
                    mlflow_tracker.log_metrics(metrics)
                    mlflow_tracker.log_param('model_type', model_type)
                except Exception as e:
                    logger.warning(f"MLFlow logging failed for {model_type}: {e}")
            
            # Log to DuckDB
            if duckdb_reporter:
                try:
                    duckdb_reporter.log_metrics(
                        experiment_id,
                        model_type,
                        None,
                        metrics
                    )
                except Exception as e:
                    logger.warning(f"DuckDB logging failed for {model_type}: {e}")
        
        # Collect CV results for statistical analysis
        all_cv_results = {}
        for model_type, result in results.get('results', {}).items():
            if 'error' not in result and 'cv_results' in result:
                cv_res = result.get('cv_results', {})
                if cv_res and 'metrics' in cv_res:
                    all_cv_results[model_type] = cv_res
        
        # Statistical analysis
        if stats_analyzer and all_cv_results:
            logger.info("Running statistical analyses...")
            plots_dir = output_dir / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            stats_results = stats_analyzer.comprehensive_analysis(
                all_cv_results,
                save_dir=str(plots_dir)
            )
            
            # Log statistical results
            if mlflow_tracker:
                try:
                    if 'anova' in stats_results:
                        anova = stats_results['anova']
                        mlflow_tracker.log_metric('anova_f_stat', anova.get('f_statistic', 0))
                        mlflow_tracker.log_metric('anova_p_value', anova.get('p_value', 1))
                except Exception as e:
                    logger.warning(f"MLFlow stats logging failed: {e}")
        
        # Generate reports
        if duckdb_reporter:
            try:
                reports_dir = output_dir / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                report = duckdb_reporter.generate_report(
                    experiment_id,
                    output_path=str(reports_dir / f"report_{experiment_id}.txt")
                )
                logger.info("Report generated")
                print("\n" + report)
            except Exception as e:
                logger.warning(f"Report generation failed: {e}")
        
        # Save results JSON
        results_file = output_dir / f"results_{experiment_id}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")
        
        logger.info("=" * 80)
        logger.info("TRAINING PIPELINE COMPLETED SUCCESSFULLY")
        logger.info("=" * 80)
        
    except Exception as e:
        logger.error(f"Training pipeline failed: {e}", exc_info=True)
        raise
    finally:
        # Cleanup
        if mlflow_tracker:
            mlflow_tracker.end_run()
        if duckdb_reporter:
            duckdb_reporter.close()


if __name__ == '__main__':
    main()

