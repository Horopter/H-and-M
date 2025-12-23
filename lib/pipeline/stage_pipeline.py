"""
Stage-based pipeline with caching and proper workflow.
Stages: Exploratory -> Statistical -> Feature Engineering -> RFE -> Training (30% train/val) -> Full Training (100% train/val)
"""
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from scipy.sparse import csr_matrix
from pathlib import Path
import polars as pl
import json
import shutil

from ..config import get_config
from ..logging.logger import get_logger
from ..data.loader import DataLoader
from ..data.splitter import TemporalSplitter
from ..features.nlp_features import NLPFeatureExtractor
from ..features.embeddings import EmbeddingExtractor
from ..features.encodings import EncodingPipeline
from ..features.feature_union import FeatureUnion
from ..data.preprocessing_pipeline import PreprocessingPipeline
from ..training.cv import CrossValidator
from ..training.grid_search import GridSearch
from ..training.trainer import Trainer
from ..checkpointing.checkpoint_manager import CheckpointManager
from ..checkpointing.state_manager import StateManager
from ..utils.validation import LeakageDetector
from ..training.mlflow_tracker import MLFlowTracker
from ..utils.duckdb_reporter import DuckDBReporter
from ..utils.stats_analysis import StatisticalAnalyzer
from ..utils.exploratory import ExploratoryAnalyzer
from ..utils.rfe import RecursiveFeatureElimination
from ..utils.visualization import Visualizer
from ..utils.arrow_storage import ArrowStorage
from ..utils.submission import SubmissionGenerator

logger = get_logger(__name__)


class StagePipeline:
    """Stage-based pipeline with caching and proper workflow."""
    
    def __init__(self, config=None):
        """Initialize stage pipeline."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        
        # Set consistent random seed
        np.random.seed(self.config.random_state)
        import random
        random.seed(self.config.random_state)
        
        # Initialize components
        self.data_loader = DataLoader(config)
        self.splitter = TemporalSplitter(config)
        # DataSplitter not needed - we use provided train/val/test files
        self.leakage_detector = LeakageDetector(config)
        self.checkpoint_manager = CheckpointManager(config)
        self.state_manager = StateManager(config)
        self.cv = CrossValidator(config)
        self.grid_search = GridSearch(config)
        
        # Feature extractors (will be fitted once and cached)
        self.nlp_extractor = NLPFeatureExtractor(config)
        self.embedding_extractor = EmbeddingExtractor(config)
        self.encoding_pipeline = EncodingPipeline()
        self.feature_union = FeatureUnion(config)
        
        # Analysis tools
        self.exploratory_analyzer = ExploratoryAnalyzer(config)
        self.stats_analyzer = StatisticalAnalyzer(config)
        self.rfe = RecursiveFeatureElimination(config)
        self.visualizer = Visualizer(config)
        self.arrow_storage = ArrowStorage(config)
        self.submission_generator = SubmissionGenerator(config)
        
        # Cache for features (avoid recomputation)
        self.feature_cache: Dict[str, Any] = {}
        self.preprocessing_cache: Dict[str, Any] = {}
        
        # Results storage
        self.stage_results: Dict[str, Any] = {}
        
        # Cleanup existing checkpoints if DELETE_EXISTING is True
        if self.config.delete_existing:
            self._cleanup_existing_checkpoints()
    
    def stage_1_exploratory(self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None) -> Dict[str, Any]:
        """Stage 1: Exploratory Data Analysis."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 1: EXPLORATORY DATA ANALYSIS")
        self.logger.info("=" * 80)
        
        results = self.exploratory_analyzer.analyze(train_df, val_df, test_df)
        
        # Save exploratory results
        output_dir = Path(self.config.output_dir) / "exploratory"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        with open(output_dir / "eda_results.json", 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        self.stage_results['exploratory'] = results
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_1', results)
        
        self.logger.info("Stage 1 complete: Exploratory analysis")
        return results
    
    def stage_2_statistical_analysis(self, train_df: pl.DataFrame, val_df: pl.DataFrame) -> Dict[str, Any]:
        """Stage 2: Statistical analyses (ANOVA, Tukey HSD, F statistic, Cramer's V, etc.)."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 2: STATISTICAL ANALYSIS")
        self.logger.info("=" * 80)
        
        results = self.stats_analyzer.comprehensive_statistical_analysis(
            train_df, val_df, save_dir=str(Path(self.config.output_dir) / "statistical")
        )
        
        self.stage_results['statistical'] = results
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_2', results)
        
        self.logger.info("Stage 2 complete: Statistical analysis")
        return results
    
    def stage_3_feature_engineering(self, train_df: pl.DataFrame, val_df: pl.DataFrame, test_df: Optional[pl.DataFrame] = None) -> Dict[str, csr_matrix]:
        """Stage 3: Feature engineering (done once, cached)."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 3: FEATURE ENGINEERING")
        self.logger.info("=" * 80)
        
        # Check cache
        cache_key = f"features_{hash(str(train_df.shape) + str(val_df.shape))}"
        if cache_key in self.feature_cache:
            self.logger.info("Using cached features")
            return self.feature_cache[cache_key]
        
        # Get texts
        train_texts = train_df[self.config.text_column].to_list()
        val_texts = val_df[self.config.text_column].to_list()
        test_texts = test_df[self.config.text_column].to_list() if test_df is not None else None
        
        # NLP features (fit once on train, transform on all)
        self.logger.info("Extracting NLP features...")
        train_nlp = self.nlp_extractor.extract_all_features(train_texts, fit=True)
        val_nlp = self.nlp_extractor.transform(val_texts)
        test_nlp = self.nlp_extractor.transform(test_texts) if test_texts else None
        
        # Embeddings (fit once on train, transform on all)
        embeddings_train = {}
        embeddings_val = {}
        embeddings_test = {}
        
        if self.config.use_embeddings:
            self.logger.info("Extracting embeddings...")
            self.embedding_extractor.initialize_embeddings()
            embeddings_train = self.embedding_extractor.extract_all_embeddings(train_texts, train_nlp)
            embeddings_val = self.embedding_extractor.extract_all_embeddings(val_texts)
            if test_texts:
                embeddings_test = self.embedding_extractor.extract_all_embeddings(test_texts)
        
        # Encodings (fit once on train, transform on all)
        encodings_train = {}
        encodings_val = {}
        encodings_test = {}
        
        if self.config.use_encodings:
            self.logger.info("Applying encodings...")
            # Fit on train only
            y_train = train_df[self.config.label_column].to_numpy()
            # Note: Encodings would be applied here if we had categorical columns
            # For now, text-based features don't need traditional encodings
        
        # Combine all features (done once)
        train_features = self.feature_union.combine_features(
            nlp_features=train_nlp,
            embeddings=embeddings_train,
            encodings=encodings_train
        )
        
        val_features = self.feature_union.combine_features(
            nlp_features=val_nlp,
            embeddings=embeddings_val,
            encodings=encodings_val
        )
        
        test_features = self.feature_union.combine_features(
            nlp_features=test_nlp,
            embeddings=embeddings_test,
            encodings=encodings_test
        ) if test_nlp else None
        
        features = {
            'train': train_features,
            'val': val_features,
            'test': test_features
        }
        
        # Cache features
        self.feature_cache[cache_key] = features
        
        # Save features to Arrow format
        self.logger.info("Saving engineered features to Arrow format...")
        self.arrow_storage.save_sparse_matrix(train_features, "train_features", "stage_3")
        self.arrow_storage.save_sparse_matrix(val_features, "val_features", "stage_3")
        if test_features is not None:
            self.arrow_storage.save_sparse_matrix(test_features, "test_features", "stage_3")
        
        # Save feature statistics
        feature_stats = {
            'train_shape_0': train_features.shape[0],
            'train_shape_1': train_features.shape[1],
            'train_nnz': train_features.nnz,
            'val_shape_0': val_features.shape[0],
            'val_shape_1': val_features.shape[1],
            'val_nnz': val_features.nnz
        }
        self.arrow_storage.save_metrics(feature_stats, "feature_statistics", "stage_3")
        
        # Save stage checkpoint
        stage_3_results = {
            'feature_shapes': {
                'train': list(train_features.shape),
                'val': list(val_features.shape),
                'test': list(test_features.shape) if test_features is not None else None
            },
            'feature_statistics': feature_stats
        }
        self._save_stage_checkpoint('stage_3', stage_3_results)
        
        self.logger.info(f"Feature engineering complete. Train: {train_features.shape}, Val: {val_features.shape}")
        self.logger.info("Stage 3 complete: Feature engineering (cached)")
        
        return features
    
    def stage_4_preprocessing(self, features: Dict[str, csr_matrix]) -> Dict[str, csr_matrix]:
        """Stage 4: Preprocessing (scaling, imputation, PCA, normalization) - done once, cached."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 4: PREPROCESSING")
        self.logger.info("=" * 80)
        
        # Check cache
        cache_key = f"preprocessing_{hash(str(features['train'].shape))}"
        if cache_key in self.preprocessing_cache:
            self.logger.info("Using cached preprocessed features")
            return self.preprocessing_cache[cache_key]
        
        # Create preprocessing pipeline
        preprocessor = PreprocessingPipeline(self.config)
        
        # Fit on train, transform on all
        processed_train = preprocessor.fit_transform(features['train'])
        processed_val = preprocessor.transform(features['val'])
        processed_test = preprocessor.transform(features['test']) if features.get('test') else None
        
        processed = {
            'train': processed_train,
            'val': processed_val,
            'test': processed_test
        }
        
        # Cache preprocessed features
        self.preprocessing_cache[cache_key] = processed
        
        # Save preprocessed features to Arrow
        self.logger.info("Saving preprocessed features to Arrow format...")
        self.arrow_storage.save_sparse_matrix(processed_train, "train_preprocessed", "stage_4")
        self.arrow_storage.save_sparse_matrix(processed_val, "val_preprocessed", "stage_4")
        if processed_test is not None:
            self.arrow_storage.save_sparse_matrix(processed_test, "test_preprocessed", "stage_4")
        
        # Save stage checkpoint
        stage_4_results = {
            'processed_shapes': {
                'train': list(processed_train.shape),
                'val': list(processed_val.shape),
                'test': list(processed_test.shape) if processed_test is not None else None
            }
        }
        self._save_stage_checkpoint('stage_4', stage_4_results)
        
        self.logger.info(f"Preprocessing complete. Train: {processed_train.shape}, Val: {processed_val.shape}")
        self.logger.info("Stage 4 complete: Preprocessing (cached)")
        
        return processed
    
    def stage_5_rfe(self, X_train: csr_matrix, y_train: np.ndarray, X_val: csr_matrix, y_val: np.ndarray) -> Dict[str, Any]:
        """Stage 5: Recursive Feature Elimination."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 5: RECURSIVE FEATURE ELIMINATION (RFE)")
        self.logger.info("=" * 80)
        
        # Use subset for RFE (30% of data)
        subset_size = 0.3
        n_subset = max(1, int(len(y_train) * subset_size))
        
        if n_subset < len(y_train):
            from sklearn.model_selection import StratifiedKFold
            skf = StratifiedKFold(n_splits=1, shuffle=True, random_state=self.config.random_state)
            indices = np.arange(len(y_train))
            train_idx, _ = next(skf.split(indices, y_train))
            subset_indices = train_idx[:n_subset]
            
            # Index sparse matrix with subset indices
            X_rfe = X_train[subset_indices]
            y_rfe = y_train[subset_indices]
        else:
            X_rfe = X_train
            y_rfe = y_train
        
        # Perform RFE
        rfe_results = self.rfe.fit_transform(
            X_rfe, y_rfe, 
            n_features_to_select=None,  # Select optimal number
            step=0.1  # Remove 10% of features at each step
        )
        
        # Apply feature selection to full datasets
        # Get processed test features if available
        processed_test = None
        for cache_key, cached_features in self.preprocessing_cache.items():
            if isinstance(cached_features, dict) and 'test' in cached_features:
                processed_test = cached_features['test']
                break
        
        selected_features = {
            'train': self.rfe.transform(X_train),
            'val': self.rfe.transform(X_val),
            'test': self.rfe.transform(processed_test) if processed_test is not None else None
        }
        
        self.stage_results['rfe'] = rfe_results
        
        # Generate RFE visualization
        plots_dir = Path(self.config.output_dir) / "plots" / "stage_5_rfe"
        plots_dir.mkdir(parents=True, exist_ok=True)
        self.visualizer.plot_rfe_results(
            rfe_results,
            save_path=str(plots_dir / "rfe_results.png")
        )
        
        # Save RFE results to Arrow
        self.arrow_storage.save_metrics(rfe_results, "rfe_results", "stage_5")
        
        # Save selected features
        self.arrow_storage.save_sparse_matrix(selected_features['train'], "train_features_selected", "stage_5")
        self.arrow_storage.save_sparse_matrix(selected_features['val'], "val_features_selected", "stage_5")
        if selected_features['test'] is not None:
            self.arrow_storage.save_sparse_matrix(selected_features['test'], "test_features_selected", "stage_5")
        
        self.logger.info(f"RFE complete. Selected {rfe_results.get('n_features_selected', 'unknown')} features")
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_5', rfe_results)
        
        self.logger.info("Stage 5 complete: RFE")
        
        return selected_features
    
    def stage_6_training_30pct(self, X_train: csr_matrix, y_train: np.ndarray, X_val: csr_matrix, y_val: np.ndarray, experiment_name: str = "default") -> Dict[str, Any]:
        """Stage 6: Training on 30% of train/val data with stratified 5-fold CV and hyperparameter tuning."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 6: TRAINING ON 30% OF TRAIN/VAL DATA (CV + HYPERPARAMETER TUNING)")
        self.logger.info("=" * 80)
        
        # Sample 30% of train and 30% of val data for CV and hyperparameter tuning
        train_sample_size = int(len(y_train) * 0.3)
        val_sample_size = int(len(y_val) * 0.3)
        
        self.logger.info(f"Sampling {train_sample_size} from {len(y_train)} train samples (30%)")
        self.logger.info(f"Sampling {val_sample_size} from {len(y_val)} val samples (30%)")
        self.logger.info("CV folds will use 10% of original data per fold (random sampling)")
        
        # Random stratified sampling of 30% from train and val
        from sklearn.model_selection import train_test_split
        
        # Sample 30% of train data (get indices first, then index sparse matrix)
        train_indices = np.arange(len(y_train))
        train_sample_indices, _ = train_test_split(
            train_indices,
            train_size=0.3,
            stratify=y_train,
            random_state=self.config.random_state
        )
        # Index sparse matrix with sample indices
        X_train_30pct = X_train[train_sample_indices]
        y_train_30pct = y_train[train_sample_indices]
        
        # Sample 30% of val data
        val_indices = np.arange(len(y_val))
        val_sample_indices, _ = train_test_split(
            val_indices,
            train_size=0.3,
            stratify=y_val,
            random_state=self.config.random_state
        )
        # Index sparse matrix with sample indices
        X_val_30pct = X_val[val_sample_indices]
        y_val_30pct = y_val[val_sample_indices]
        
        # For CV: use 10% of original data per fold
        cv_sample_per_fold = 0.1  # 10% of original data per fold
        
        all_results = {}
        
        for model_type in self.config.models:
            try:
                self.logger.info(f"Training {model_type} on 30% subset...")
                
                # Create model factory
                model_factory = self._get_model_factory(model_type)
                
                # Grid search on 30% train data, using 10% of original data per fold
                param_grid = self.config.hyperparameter_grids.get(model_type, {})
                if param_grid:
                    # Grid search will use CV internally, which will sample 10% per fold
                    grid_results = self.grid_search.search(
                        model_factory,
                        param_grid,
                        X_train_30pct,
                        y_train_30pct,
                        subset_size=cv_sample_per_fold,  # 10% of original data per fold
                        cv_folds=self.config.cv_folds,
                        scoring='f1',
                        original_data_size=self.original_data_size
                    )
                    best_params = grid_results.get('best_params', {})
                else:
                    best_params = {}
                    grid_results = {}
                
                # Cross-validation on 30% data: use 10% of original data per fold (random sampling)
                cv_results = self.cv.evaluate_model(
                    model_factory,
                    X_train_30pct,
                    y_train_30pct,
                    subset_size=cv_sample_per_fold,  # 10% of original data per fold
                    temporal=False,  # Random sampling, not temporal
                    model_params=best_params,
                    save_checkpoints=False,  # Don't save during 30% training
                    original_data_size=self.original_data_size
                )
                
                all_results[model_type] = {
                    'cv_results': cv_results,
                    'grid_search': grid_results,
                    'best_params': best_params
                }
                
                self.logger.info(f"{model_type} CV F1: {cv_results.get('metrics', {}).get('f1', {}).get('mean', 0):.4f}")
                
            except Exception as e:
                self.logger.error(f"Error training {model_type}: {e}", exc_info=True)
                all_results[model_type] = {'error': str(e)}
        
        # Statistical analysis of CV results
        if self.stats_analyzer:
            self.logger.info("Performing statistical analysis on CV results...")
            plots_dir = Path(self.config.output_dir) / "plots" / "stage_6_training"
            plots_dir.mkdir(parents=True, exist_ok=True)
            
            stats_results = self.stats_analyzer.comprehensive_analysis(
                all_results,
                save_dir=str(plots_dir)
            )
            all_results['_statistical_analysis'] = stats_results
        
        # Generate comprehensive visualizations
        self.logger.info("Generating training visualizations...")
        plots_dir = Path(self.config.output_dir) / "plots" / "stage_6_training"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # CV metrics plot
        self.visualizer.plot_cv_metrics(
            all_results,
            save_path=str(plots_dir / "cv_metrics.png")
        )
        
        # Model comparison
        model_metrics = {}
        for model_type, result in all_results.items():
            if 'error' not in result and 'cv_results' in result:
                metrics = result['cv_results'].get('metrics', {})
                model_metrics[model_type] = {
                    k: v.get('mean', 0) for k, v in metrics.items() if isinstance(v, dict)
                }
        
        if model_metrics:
            self.visualizer.plot_model_comparison(
                model_metrics,
                save_path=str(plots_dir / "model_comparison.png")
            )
        
        # Save CV results to Arrow
        self.arrow_storage.save_cv_results(all_results, "training_30pct", "stage_6")
        
        # Save metrics
        for model_type, result in all_results.items():
            if 'error' not in result:
                self.arrow_storage.save_metrics(result, f"{model_type}_30pct", "stage_6")
        
        # Find best model
        best_model_type = None
        best_f1 = -1
        
        for model_type, result in all_results.items():
            if 'error' not in result and 'cv_results' in result:
                cv_metrics = result['cv_results'].get('metrics', {})
                f1_mean = cv_metrics.get('f1', {}).get('mean', 0)
                if f1_mean > best_f1:
                    best_f1 = f1_mean
                    best_model_type = model_type
        
        self.stage_results['training_30pct'] = all_results
        self.stage_results['best_model'] = best_model_type
        self.stage_results['best_f1'] = best_f1
        
        # Save stage checkpoint
        stage_6_checkpoint = {
            'all_results': all_results,
            'best_model': best_model_type,
            'best_f1': best_f1
        }
        self._save_stage_checkpoint('stage_6', stage_6_checkpoint)
        
        if best_model_type:
            self.logger.info(f"Stage 6 complete: Best model is {best_model_type} with F1={best_f1:.4f}")
        else:
            self.logger.warning("Stage 6 complete: No best model found (all models may have failed)")
        
        return all_results
    
    def stage_7_full_training(self, X_train: csr_matrix, y_train: np.ndarray, X_val: csr_matrix, y_val: np.ndarray, best_model_type: str, best_params: Dict[str, Any], experiment_name: str = "default") -> Dict[str, Any]:
        """Stage 7: Train best model on ALL train and ALL val data."""
        self.logger.info("=" * 80)
        self.logger.info("STAGE 7: TRAINING BEST MODEL ON ALL TRAIN/VAL DATA")
        self.logger.info("=" * 80)
        self.logger.info(f"Best model: {best_model_type} with params: {best_params}")
        self.logger.info(f"Using ALL train data: {len(y_train)} samples")
        self.logger.info(f"Using ALL val data: {len(y_val)} samples")
        
        # Create best model with best params
        model_factory = self._get_model_factory(best_model_type)
        model = model_factory(**best_params)
        
        # Train on ALL train data with 5-fold CV (for model selection and saving per fold)
        # Each fold uses 10% of original data (random sampling)
        original_size = getattr(self, 'original_data_size', len(y_train) * 5)
        cv_sample_per_fold = 0.1  # 10% of original data per fold
        
        cv_results = self.cv.evaluate_model(
            model_factory,
            X_train,
            y_train,
            subset_size=cv_sample_per_fold,  # 10% of original data per fold
            temporal=False,  # Random sampling
            model_params=best_params,
            save_checkpoints=True,  # Save models for each fold
            original_data_size=self.original_data_size
        )
        
        # Also train final model on all training data
        final_model = model_factory(**best_params)
        final_model.fit(X_train, y_train)
        
        # Final evaluation on validation set
        model = final_model
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, roc_auc_score
        
        final_metrics = {
            'f1': f1_score(y_val, y_pred),
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred, zero_division=0),
            'recall': recall_score(y_val, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_val, y_proba) if y_proba is not None else 0.0
        }
        
        # Generate comprehensive visualizations for final model
        self.logger.info("Generating final model visualizations...")
        plots_dir = Path(self.config.output_dir) / "plots" / "stage_7_final"
        plots_dir.mkdir(parents=True, exist_ok=True)
        
        # Confusion matrix
        self.visualizer.plot_confusion_matrix(
            y_val, y_pred, best_model_type,
            save_path=str(plots_dir / "confusion_matrix.png")
        )
        
        # ROC curve
        if y_proba is not None:
            self.visualizer.plot_roc_curve(
                y_val, y_proba, best_model_type,
                save_path=str(plots_dir / "roc_curve.png")
            )
            
            # Precision-Recall curve
            self.visualizer.plot_precision_recall_curve(
                y_val, y_proba, best_model_type,
                save_path=str(plots_dir / "precision_recall_curve.png")
            )
        
        # Save predictions and probabilities to Arrow
        pred_df = pl.DataFrame({
            'y_true': y_val,
            'y_pred': y_pred,
            'y_proba': y_proba if y_proba is not None else [0.0] * len(y_val)
        })
        self.arrow_storage.save_dataframe(pred_df, "final_predictions", "stage_7")
        
        # Save final metrics
        self.arrow_storage.save_metrics(final_metrics, "final_metrics", "stage_7")
        self.arrow_storage.save_metrics(cv_results, "final_cv_results", "stage_7")
        
        # Save final model
        checkpoint_path = self.checkpoint_manager.save_checkpoint(
            model,
            best_model_type,
            fold_id=None,
            score=final_metrics['f1'],
            metadata={'stage': 'final', 'params': best_params, 'metrics': final_metrics}
        )
        
        results = {
            'model_type': best_model_type,
            'params': best_params,
            'cv_results': cv_results,
            'final_metrics': final_metrics,
            'checkpoint_path': str(checkpoint_path)
        }
        
        self.stage_results['full_training'] = results
        
        # Save stage checkpoint
        self._save_stage_checkpoint('stage_7', results)
        
        self.logger.info(f"Stage 7 complete: Final model F1={final_metrics['f1']:.4f}")
        
        return results
    
    def _get_model_factory(self, model_type: str):
        """Get model factory function."""
        from ..models.logistic_regression import LogisticRegressionModel
        from ..models.svm import SVMModel
        from ..models.bayesian import BayesianModel
        from ..models.xgboost import XGBoostModel
        from ..models.neural_network import NeuralNetworkModel
        
        factories = {
            'logreg': lambda **kwargs: LogisticRegressionModel(self.config, **kwargs),
            'svm': lambda **kwargs: SVMModel(self.config, **kwargs),
            'bayesian': lambda **kwargs: BayesianModel(self.config, **kwargs),
            'xgboost': lambda **kwargs: XGBoostModel(self.config, **kwargs),
            'neural_net': lambda **kwargs: NeuralNetworkModel(self.config, **kwargs)
        }
        
        if model_type not in factories:
            raise ValueError(f"Unknown model type: {model_type}")
        
        return factories[model_type]
    
    def _cleanup_existing_checkpoints(self):
        """Delete existing checkpoints and stage results if DELETE_EXISTING is True."""
        self.logger.info("DELETE_EXISTING is True - cleaning up existing checkpoints and stage results")
        
        # Clean up checkpoint directory
        checkpoint_dir = Path(self.config.checkpoint_dir)
        if checkpoint_dir.exists():
            try:
                shutil.rmtree(checkpoint_dir)
                self.logger.info(f"Deleted checkpoint directory: {checkpoint_dir}")
            except Exception as e:
                self.logger.warning(f"Could not delete checkpoint directory {checkpoint_dir}: {e}")
        
        # Clean up Arrow storage (stage results)
        arrow_storage_dir = Path(self.config.output_dir) / "arrow_data"
        if arrow_storage_dir.exists():
            try:
                shutil.rmtree(arrow_storage_dir)
                self.logger.info(f"Deleted Arrow storage directory: {arrow_storage_dir}")
            except Exception as e:
                self.logger.warning(f"Could not delete Arrow storage directory {arrow_storage_dir}: {e}")
        
        # Clean up stage checkpoint JSON files
        checkpoint_json_dir = Path(self.config.output_dir) / "checkpoints"
        if checkpoint_json_dir.exists():
            try:
                for json_file in checkpoint_json_dir.glob("*_results.json"):
                    json_file.unlink()
                    self.logger.debug(f"Deleted checkpoint JSON: {json_file}")
            except Exception as e:
                self.logger.warning(f"Could not delete checkpoint JSON files: {e}")
        
        # Recreate directories
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        arrow_storage_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_json_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Cleanup complete - ready for fresh run")
    
    def _save_stage_checkpoint(self, stage_name: str, stage_results: Dict[str, Any]):
        """Save stage results as checkpoint."""
        try:
            # Save stage results to Arrow storage
            self.arrow_storage.save_metrics(
                stage_results,
                f"{stage_name}_checkpoint",
                stage_name
            )
            
            # Also save to JSON for human readability
            checkpoint_json_path = Path(self.config.output_dir) / "checkpoints" / f"{stage_name}_results.json"
            checkpoint_json_path.parent.mkdir(parents=True, exist_ok=True)
            with open(checkpoint_json_path, 'w') as f:
                json.dump(stage_results, f, indent=2, default=str)
            
            self.logger.info(f"Saved checkpoint for {stage_name}")
        except Exception as e:
            self.logger.warning(f"Could not save checkpoint for {stage_name}: {e}")
    
    def run_all_stages(self, experiment_name: str = "default", mlflow_tracker: Optional[Any] = None, duckdb_reporter: Optional[Any] = None) -> Dict[str, Any]:
        """Run all pipeline stages in sequence."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING STAGE-BASED PIPELINE")
        self.logger.info("=" * 80)
        self.logger.info(f"Random seed: {self.config.random_state} (consistent across all stages)")
        
        # Load train/val/test files (they already exist, no splitting needed)
        train_df, val_df, test_df = self.data_loader.load_train_val_test()
        self.logger.info("Loaded train/val/test files")
        
        # Store original sizes for CV sampling (10% of original data per fold)
        self.original_train_size = len(train_df)
        self.original_val_size = len(val_df)
        self.original_data_size = len(train_df) + len(val_df) + len(test_df)
        self.logger.info(f"Original data sizes - Train: {self.original_train_size}, Val: {self.original_val_size}, Test: {len(test_df)}, Total: {self.original_data_size}")
        
        # Stage 1: Exploratory
        exploratory_results = self.stage_1_exploratory(train_df, val_df, test_df)
        
        # Stage 2: Statistical analysis
        statistical_results = self.stage_2_statistical_analysis(train_df, val_df)
        
        # Save statistical results to Arrow
        if statistical_results:
            self.arrow_storage.save_metrics(statistical_results, "statistical_analysis", "stage_2")
        
        # Stage 3: Feature engineering (cached)
        features = self.stage_3_feature_engineering(train_df, val_df, test_df)
        
        # Stage 4: Preprocessing (cached)
        processed_features = self.stage_4_preprocessing(features)
        
        # Get targets
        y_train = train_df[self.config.label_column].to_numpy()
        y_val = val_df[self.config.label_column].to_numpy()
        
        # Stage 5: RFE (optional, can be skipped)
        if self.config.use_rfe:
            rfe_features = self.stage_5_rfe(
                processed_features['train'],
                y_train,
                processed_features['val'],
                y_val
            )
            # Update processed_features with RFE-selected features
            processed_features = rfe_features
        
        # Stage 6: Training on 30% of train/val with CV and hyperparameter tuning
        training_30pct_results = self.stage_6_training_30pct(
            processed_features['train'],
            y_train,
            processed_features['val'],
            y_val,
            experiment_name
        )
        
        # Stage 7: Train best model on ALL train/val data
        best_model_type = self.stage_results.get('best_model')
        best_params = training_30pct_results.get(best_model_type, {}).get('best_params', {})
        
        if best_model_type:
            full_training_results = self.stage_7_full_training(
                processed_features['train'],
                y_train,
                processed_features['val'],
                y_val,
                best_model_type,
                best_params,
                experiment_name
            )
        else:
            self.logger.error("No best model found, skipping full training")
            full_training_results = {}
        
        # Generate submission file if test data is available
        submission_path = None
        if test_df is not None and processed_features.get('test') is not None and best_model_type:
            self.logger.info("Generating submission file...")
            try:
                # Get test IDs
                test_ids = test_df[self.config.id_column].to_numpy() if self.config.id_column in test_df.columns else None
                
                # Make predictions on test set using best model
                final_model = self._get_model_factory(best_model_type)(**best_params)
                final_model.fit(processed_features['train'], y_train)
                test_predictions = final_model.predict_proba(processed_features['test'])[:, 1] if hasattr(final_model, 'predict_proba') else final_model.predict(processed_features['test'])
                
                # Generate submission CSV (ONLY CSV file)
                submission_path = self.submission_generator.generate_submission(
                    test_predictions,
                    test_ids
                )
                
                # Save test predictions to Arrow (not CSV)
                test_pred_df = pl.DataFrame({
                    'id': test_ids if test_ids is not None else np.arange(len(test_predictions)),
                    'prediction': test_predictions
                })
                self.arrow_storage.save_dataframe(test_pred_df, "test_predictions", "submission")
                
            except Exception as e:
                self.logger.warning(f"Could not generate submission file: {e}")
        
        # Compile final results
        final_results = {
            'exploratory': exploratory_results,
            'statistical': statistical_results,
            'feature_engineering': {
                'train_shape': processed_features['train'].shape,
                'val_shape': processed_features['val'].shape
            },
            'training_30pct': training_30pct_results,
            'full_training': full_training_results,
            'best_model': best_model_type,
            'best_f1': self.stage_results.get('best_f1'),
            'submission_path': str(submission_path) if submission_path else None
        }
        
        # Log to MLFlow and DuckDB
        if mlflow_tracker:
            mlflow_tracker.log_metrics(final_results.get('full_training', {}).get('final_metrics', {}))
            mlflow_tracker.log_param('best_model', best_model_type or 'none')
        
        if duckdb_reporter:
            duckdb_reporter.log_cv_results(experiment_name, best_model_type or 'unknown', None, training_30pct_results.get(best_model_type, {}).get('cv_results', {}))
        
        self.logger.info("=" * 80)
        self.logger.info("PIPELINE COMPLETE")
        self.logger.info("=" * 80)
        
        return final_results

