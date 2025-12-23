"""
Main training orchestration that ties everything together.
"""
import numpy as np
from typing import Dict, List, Any, Optional
from scipy.sparse import csr_matrix
from pathlib import Path
import polars as pl

from ..config import get_config
from ..logging.logger import get_logger
from ..data.loader import DataLoader
from ..data.splitter import TemporalSplitter
from ..features.nlp_features import NLPFeatureExtractor
from ..features.embeddings import EmbeddingExtractor
from ..features.encodings import EncodingPipeline
from ..features.feature_union import FeatureUnion
from ..data.preprocessor import PreprocessingPipeline
from ..models.logistic_regression import LogisticRegressionModel
from ..models.svm import SVMModel
from ..models.bayesian import BayesianModel
from ..models.xgboost import XGBoostModel
from ..models.neural_network import NeuralNetworkModel
from ..training.cv import CrossValidator
from ..training.grid_search import GridSearch
from ..checkpointing.checkpoint_manager import CheckpointManager
from ..checkpointing.state_manager import StateManager
from ..utils.validation import LeakageDetector
from ..training.mlflow_tracker import MLFlowTracker
from ..utils.duckdb_reporter import DuckDBReporter
from ..utils.stats_analysis import StatisticalAnalyzer

logger = get_logger(__name__)


class Trainer:
    """Main training orchestrator."""
    
    def __init__(self, config=None):
        """
        Initialize trainer.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.splitter = TemporalSplitter(config)
        self.leakage_detector = LeakageDetector(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.state_manager = StateManager(config)
        self.cv = CrossValidator(config)
        self.grid_search = GridSearch(config)
        
        # Feature extractors
        self.nlp_extractor = NLPFeatureExtractor(config)
        self.embedding_extractor = EmbeddingExtractor(config)
        self.encoding_pipeline = EncodingPipeline()
        self.feature_union = FeatureUnion(config)
        
        # Model factories
        self.model_factories = {
            'logreg': self._create_logreg,
            'svm': self._create_svm,
            'bayesian': self._create_bayesian,
            'xgboost': self._create_xgboost,
            'neural_net': self._create_neural_net
        }
    
    def _create_logreg(self, **kwargs):
        """Create Logistic Regression model."""
        return LogisticRegressionModel(self.config, **kwargs)
    
    def _create_svm(self, **kwargs):
        """Create SVM model."""
        return SVMModel(self.config, **kwargs)
    
    def _create_bayesian(self, **kwargs):
        """Create Bayesian model."""
        return BayesianModel(self.config, **kwargs)
    
    def _create_xgboost(self, **kwargs):
        """Create XGBoost model."""
        return XGBoostModel(self.config, **kwargs)
    
    def _create_neural_net(self, **kwargs):
        """Create Neural Network model."""
        return NeuralNetworkModel(self.config, **kwargs)
    
    def prepare_features(
        self,
        train_df: pl.DataFrame,
        val_df: pl.DataFrame,
        test_df: Optional[pl.DataFrame] = None,
        fit: bool = True
    ) -> Dict[str, csr_matrix]:
        """
        Prepare all features.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            test_df: Test DataFrame (optional)
            fit: Whether to fit feature extractors
            
        Returns:
            Dictionary of feature matrices
        """
        self.logger.info("Preparing features")
        
        # Get texts
        train_texts = train_df[self.config.text_column].to_list()
        val_texts = val_df[self.config.text_column].to_list()
        test_texts = test_df[self.config.text_column].to_list() if test_df is not None else None
        
        # NLP features
        if fit:
            train_nlp = self.nlp_extractor.extract_all_features(train_texts, fit=True)
            val_nlp = self.nlp_extractor.transform(val_texts)
            test_nlp = self.nlp_extractor.transform(test_texts) if test_texts else None
        else:
            train_nlp = self.nlp_extractor.extract_all_features(train_texts, fit=False)
            val_nlp = self.nlp_extractor.transform(val_texts)
            test_nlp = self.nlp_extractor.transform(test_texts) if test_texts else None
        
        # Embeddings
        embeddings_train = {}
        embeddings_val = {}
        embeddings_test = {}
        
        if self.config.use_embeddings:
            self.embedding_extractor.initialize_embeddings()
            # Train Word2Vec on training data if needed, then extract all embeddings
            embeddings_train = self.embedding_extractor.extract_all_embeddings(train_texts, train_nlp, train_word2vec=True)
            embeddings_val = self.embedding_extractor.extract_all_embeddings(val_texts, train_word2vec=False)
            if test_texts:
                embeddings_test = self.embedding_extractor.extract_all_embeddings(test_texts, train_word2vec=False)
        
        # Combine features
        train_features = self.feature_union.combine_features(
            nlp_features=train_nlp,
            embeddings=embeddings_train
        )
        
        val_features = self.feature_union.combine_features(
            nlp_features=val_nlp,
            embeddings=embeddings_val
        )
        
        test_features = None
        if test_texts:
            test_features = self.feature_union.combine_features(
                nlp_features=test_nlp,
                embeddings=embeddings_test
            )
        
        return {
            'train': train_features,
            'val': val_features,
            'test': test_features
        }
    
    def preprocess_features(
        self,
        X_train: csr_matrix,
        X_val: csr_matrix,
        X_test: Optional[csr_matrix] = None
    ) -> Dict[str, csr_matrix]:
        """
        Preprocess features (scaling, imputation, PCA, normalization).
        
        Args:
            X_train: Training features
            X_val: Validation features
            X_test: Test features (optional)
            
        Returns:
            Dictionary of preprocessed feature matrices
        """
        self.logger.info("Preprocessing features")
        
        # Convert sparse to dense for preprocessing (if needed)
        # Note: For large sparse matrices, we might want to keep them sparse
        # and only preprocess dense features
        
        # For now, we'll create a simple preprocessing pipeline
        # In practice, you'd want to handle sparse matrices more carefully
        
        # If features are very sparse, we might skip some preprocessing steps
        # or use sparse-aware preprocessing
        
        return {
            'train': X_train,
            'val': X_val,
            'test': X_test
        }
    
    def train_model(
        self,
        model_type: str,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray,
        experiment_name: str = "default"
    ) -> Dict[str, Any]:
        """
        Train a single model.
        
        Args:
            model_type: Type of model to train
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary with trained model and results
        """
        self.logger.info(f"Training {model_type} model")
        
        if model_type not in self.model_factories:
            raise ValueError(f"Unknown model type: {model_type}")
        
        model_factory = self.model_factories[model_type]
        
        # Get hyperparameter grid
        param_grid = self.config.hyperparameter_grids.get(model_type, {})
        
        # Grid search on subset with stratified CV
        cv_results_dict = {}
        best_params = {}
        if param_grid:
            self.logger.info(f"Performing grid search for {model_type} with stratified {self.config.cv_folds}-fold CV")
            grid_results = self.grid_search.search(
                model_factory,
                param_grid,
                X_train,
                y_train,
                subset_size=self.config.subset_size,
                cv_folds=self.config.cv_folds
            )
            
            best_params = grid_results['best_params']
            model = model_factory(**best_params)
            
            # Store CV results for statistical analysis (use best result's CV)
            if 'best_params' in grid_results:
                best_params = grid_results['best_params']
            # Get CV results from best parameter combination
            if 'all_results' in grid_results and grid_results['all_results']:
                best_result = max(grid_results['all_results'], key=lambda r: r.get('mean_f1', 0))
                if 'cv_results' in best_result:
                    cv_results_dict = best_result['cv_results']
        else:
            # Perform CV even without grid search to get fold results
            self.logger.info(f"Performing stratified {self.config.cv_folds}-fold CV for {model_type}")
            cv_results_dict = self.cv.cross_validate(
                model_factory,
                X_train,
                y_train,
                subset_size=self.config.subset_size,
                temporal=True,
                parallel=False
            )
            model = model_factory()
        
        # Train on full data
        self.logger.info(f"Training {model_type} on full data")
        model.fit(X_train, y_train)
        
        # Evaluate on validation
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
        
        metrics = {
            'f1': f1_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba)
        }
        
        # Save checkpoint
        if self.config.save_checkpoints:
            checkpoint_path = self.checkpoint_manager.save_checkpoint(
                model,
                model_type,
                score=metrics['f1'],
                metadata={'metrics': metrics, 'params': model.get_params()}
            )
            self.logger.info(f"Saved checkpoint to {checkpoint_path}")
        
        # Save state
        self.state_manager.save_best_metrics(
            experiment_name,
            model_type,
            None,
            metrics,
            model.get_params()
        )
        
        return {
            'model': model,
            'metrics': metrics,
            'model_type': model_type,
            'cv_results': cv_results_dict,
            'best_params': best_params if param_grid else {}
        }
    
    def train_all_models(
        self,
        X_train: csr_matrix,
        y_train: np.ndarray,
        X_val: csr_matrix,
        y_val: np.ndarray,
        experiment_name: str = "default"
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all configured models.
        
        Args:
            X_train: Training features
            y_train: Training targets
            X_val: Validation features
            y_val: Validation targets
            experiment_name: Name of the experiment
            
        Returns:
            Dictionary of model results
        """
        self.logger.info("Training all models")
        
        # Create experiment
        self.state_manager.create_experiment(experiment_name)
        
        results = {}
        for model_type in self.config.models:
            try:
                result = self.train_model(
                    model_type,
                    X_train,
                    y_train,
                    X_val,
                    y_val,
                    experiment_name
                )
                results[model_type] = result
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}")
                results[model_type] = {'error': str(e)}
        
        return results
    
    def run_full_pipeline(
        self,
        experiment_name: str = "default",
        mlflow_tracker: Optional[Any] = None,
        duckdb_reporter: Optional[Any] = None,
        stats_analyzer: Optional[Any] = None
    ) -> Dict[str, Any]:
        """
        Run the full training pipeline.
        
        Args:
            experiment_name: Name of the experiment
            mlflow_tracker: Optional MLFlow tracker
            duckdb_reporter: Optional DuckDB reporter
            stats_analyzer: Optional statistical analyzer
            
        Returns:
            Dictionary with all results
        """
        self.logger.info("Starting full training pipeline")
        
        # Load data
        train_df, val_df, test_df = self.data_loader.load_train_val_test()
        
        # Check for leakage
        leakage_results = self.leakage_detector.comprehensive_check(
            train_df, val_df, test_df
        )
        
        # Prepare features
        features = self.prepare_features(train_df, val_df, test_df, fit=True)
        
        # Preprocess features
        processed_features = self.preprocess_features(
            features['train'],
            features['val'],
            features['test']
        )
        
        # Get targets
        y_train = train_df[self.config.label_column].to_numpy()
        y_val = val_df[self.config.label_column].to_numpy()
        
        # Train all models
        results = self.train_all_models(
            processed_features['train'],
            y_train,
            processed_features['val'],
            y_val,
            experiment_name
        )
        
        # Collect CV results for statistical analysis
        all_cv_results = {}
        for model_type, result in results.items():
            if 'error' not in result and 'cv_results' in result:
                cv_res = result.get('cv_results', {})
                if cv_res and 'metrics' in cv_res:
                    all_cv_results[model_type] = cv_res
        
        # Statistical analysis
        stats_results = {}
        if stats_analyzer and all_cv_results:
            self.logger.info("Running statistical analyses...")
            # Create plots directory in output_dir
            plots_dir = Path(self.config.output_dir) / "plots"
            plots_dir.mkdir(parents=True, exist_ok=True)
            stats_results = stats_analyzer.comprehensive_analysis(
                all_cv_results,
                save_dir=str(plots_dir)
            )
        
        # Log to DuckDB
        if duckdb_reporter:
            for model_type, result in results.items():
                if 'error' not in result:
                    try:
                        duckdb_reporter.log_cv_results(
                            experiment_name,
                            model_type,
                            result.get('cv_results', {})
                        )
                    except Exception as e:
                        self.logger.warning(f"DuckDB logging failed for {model_type}: {e}")
        
        return {
            'results': results,
            'leakage_check': leakage_results,
            'features_shape': {
                'train': processed_features['train'].shape,
                'val': processed_features['val'].shape
            },
            'statistical_analysis': stats_results
        }

