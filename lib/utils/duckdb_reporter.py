"""
DuckDB integration for results storage and reporting.
"""
import json
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

try:
    import duckdb
    DUCKDB_AVAILABLE = True
except ImportError:
    DUCKDB_AVAILABLE = False

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class DuckDBReporter:
    """DuckDB-based results storage and reporting."""
    
    def __init__(self, config=None, db_path: str = "results.duckdb"):
        """
        Initialize DuckDB reporter.
        
        Args:
            config: Configuration object
            db_path: Path to DuckDB database
        """
        self.config = config or get_config()
        self.db_path = db_path
        self.logger = get_logger(self.__class__.__name__)
        
        if not DUCKDB_AVAILABLE:
            self.logger.warning("DuckDB not available, reporting disabled")
            self.enabled = False
            return
        
        self.enabled = True
        self.conn = duckdb.connect(db_path)
        self._initialize_schema()
    
    def _initialize_schema(self):
        """Initialize database schema."""
        if not self.enabled:
            return
        
        # Create experiments table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS experiments (
                experiment_id VARCHAR PRIMARY KEY,
                experiment_name VARCHAR,
                created_at TIMESTAMP,
                config JSON
            )
        """)
        
        # Create model_results table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS model_results (
                result_id INTEGER PRIMARY KEY,
                experiment_id VARCHAR,
                model_type VARCHAR,
                fold_id INTEGER,
                metric_name VARCHAR,
                metric_value DOUBLE,
                epoch INTEGER,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        # Create cv_results table
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS cv_results (
                cv_id INTEGER PRIMARY KEY,
                experiment_id VARCHAR,
                model_type VARCHAR,
                fold_id INTEGER,
                f1_score DOUBLE,
                accuracy DOUBLE,
                precision DOUBLE,
                recall DOUBLE,
                roc_auc DOUBLE,
                timestamp TIMESTAMP,
                FOREIGN KEY (experiment_id) REFERENCES experiments(experiment_id)
            )
        """)
        
        self.logger.info("DuckDB schema initialized")
    
    def log_experiment(
        self,
        experiment_id: str,
        experiment_name: str,
        config: Dict[str, Any]
    ):
        """Log experiment metadata."""
        if not self.enabled:
            return
        
        self.conn.execute("""
            INSERT OR REPLACE INTO experiments (experiment_id, experiment_name, created_at, config)
            VALUES (?, ?, ?, ?)
        """, [experiment_id, experiment_name, datetime.now(), json.dumps(config)])
        
        self.logger.info(f"Logged experiment: {experiment_id}")

    def _ensure_experiment(
        self,
        experiment_id: str,
        experiment_name: Optional[str] = None,
        config: Optional[Dict[str, Any]] = None
    ):
        """Ensure experiment exists to satisfy foreign key constraints."""
        if not self.enabled:
            return
        exists = self.conn.execute(
            "SELECT 1 FROM experiments WHERE experiment_id = ? LIMIT 1",
            [experiment_id]
        ).fetchone()
        if exists:
            return
        name = experiment_name or experiment_id
        cfg = config or self.config.to_dict()
        self.logger.warning(
            "Experiment %s not found; creating placeholder entry for logging.",
            experiment_id
        )
        self.log_experiment(experiment_id, name, cfg)

    def _next_id(self, table: str, id_column: str) -> int:
        """Get next integer ID for a table primary key."""
        row = self.conn.execute(
            f"SELECT COALESCE(MAX({id_column}), 0) + 1 FROM {table}"
        ).fetchone()
        return int(row[0]) if row else 1
    
    def log_cv_results(
        self,
        experiment_id: str,
        model_type: str,
        cv_results: Dict[str, Any]
    ):
        """Log cross-validation results."""
        if not self.enabled:
            return
        self._ensure_experiment(experiment_id)
        
        fold_results = cv_results.get('fold_results', {})
        next_id = self._next_id("cv_results", "cv_id")
        
        for fold_id, metrics in fold_results.items():
            self.conn.execute("""
                INSERT INTO cv_results (
                    cv_id, experiment_id, model_type, fold_id,
                    f1_score, accuracy, precision, recall, roc_auc, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                experiment_id,
                model_type,
                int(fold_id),
                metrics.get('f1', 0.0),
                metrics.get('accuracy', 0.0),
                metrics.get('precision', 0.0),
                metrics.get('recall', 0.0),
                metrics.get('roc_auc', 0.0),
                datetime.now()
            ])
            next_id += 1
        
        self.logger.info(f"Logged CV results for {model_type}")
    
    def log_metrics(
        self,
        experiment_id: str,
        model_type: str,
        fold_id: Optional[int],
        metrics: Dict[str, float],
        epoch: Optional[int] = None
    ):
        """Log metrics."""
        if not self.enabled:
            return
        self._ensure_experiment(experiment_id)
        next_id = self._next_id("model_results", "result_id")
        
        for metric_name, metric_value in metrics.items():
            self.conn.execute("""
                INSERT INTO model_results (
                    result_id, experiment_id, model_type, fold_id, metric_name,
                    metric_value, epoch, timestamp
                )
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, [
                next_id,
                experiment_id,
                model_type,
                fold_id,
                metric_name,
                float(metric_value),
                epoch,
                datetime.now()
            ])
            next_id += 1
    
    def generate_report(
        self,
        experiment_id: Optional[str] = None,
        output_path: Optional[str] = None
    ) -> str:
        """
        Generate comprehensive report.
        
        Args:
            experiment_id: Optional experiment ID to filter
            output_path: Optional path to save report
            
        Returns:
            Report as string
        """
        if not self.enabled:
            return "DuckDB not available"
        
        report_lines = []
        report_lines.append("=" * 80)
        report_lines.append("EXPERIMENT RESULTS REPORT")
        report_lines.append("=" * 80)
        report_lines.append("")
        
        # Summary statistics
        if experiment_id:
            query = """
                SELECT 
                    model_type,
                    AVG(f1_score) as mean_f1,
                    STDDEV(f1_score) as std_f1,
                    AVG(accuracy) as mean_acc,
                    AVG(precision) as mean_prec,
                    AVG(recall) as mean_rec,
                    AVG(roc_auc) as mean_auc
                FROM cv_results
                WHERE experiment_id = ?
                GROUP BY model_type
                ORDER BY mean_f1 DESC
            """
            results = self.conn.execute(query, [experiment_id]).fetchall()
        else:
            query = """
                SELECT 
                    model_type,
                    AVG(f1_score) as mean_f1,
                    STDDEV(f1_score) as std_f1,
                    AVG(accuracy) as mean_acc,
                    AVG(precision) as mean_prec,
                    AVG(recall) as mean_rec,
                    AVG(roc_auc) as mean_auc
                FROM cv_results
                GROUP BY model_type
                ORDER BY mean_f1 DESC
            """
            results = self.conn.execute(query).fetchall()
        
        report_lines.append("Model Performance Summary:")
        report_lines.append("-" * 80)
        report_lines.append(
            f"{'Model':<20} {'F1 (mean±std)':<20} {'Accuracy':<12} "
            f"{'Precision':<12} {'Recall':<12} {'ROC-AUC':<12}"
        )
        report_lines.append("-" * 80)
        
        for row in results:
            model_type, mean_f1, std_f1, mean_acc, mean_prec, mean_rec, mean_auc = row
            report_lines.append(
                f"{model_type:<20} {mean_f1:.4f}±{std_f1:.4f}  "
                f"{mean_acc:.4f}      {mean_prec:.4f}      "
                f"{mean_rec:.4f}      {mean_auc:.4f}"
            )
        
        report = "\n".join(report_lines)
        
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)
            self.logger.info(f"Report saved to {output_path}")
        
        return report
    
    def close(self):
        """Close database connection."""
        if self.enabled:
            self.conn.close()
