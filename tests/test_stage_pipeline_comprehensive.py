"""
Comprehensive tests for ALL functions in stage_pipeline.py.
"""
import unittest
import sys
from pathlib import Path
import numpy as np
import polars as pl
from scipy.sparse import csr_matrix
from unittest.mock import Mock, MagicMock, patch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from lib.pipeline.stage_pipeline import StagePipeline
from lib.config import get_config


class TestStagePipelineComprehensive(unittest.TestCase):
    """Comprehensive tests for all StagePipeline methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.config = get_config()
        # Mock config to avoid file dependencies
        self.config.data_path = "/tmp/test_data"
        self.config.output_dir = "/tmp/test_output"
        self.config.checkpoint_dir = "/tmp/test_checkpoints"
        self.config.train_file = "train.csv"
        self.config.val_file = "val.csv"
        self.config.test_file = "test.csv"
        self.config.text_column = "text"
        self.config.label_column = "label"
        self.config.id_column = "id"
        self.config.chunk_size = 1000
        self.config.use_embeddings = False
        self.config.use_rfe = False
        self.config.delete_existing = False
        self.config.defer_embedding_union = True
        self.config.models = ['logreg']
        
        # Create mock dataframes
        self.train_df = pl.DataFrame({
            'text': ['text1', 'text2', 'text3'],
            'label': [0, 1, 0],
            'id': [1, 2, 3]
        })
        self.val_df = pl.DataFrame({
            'text': ['text4', 'text5'],
            'label': [1, 0],
            'id': [4, 5]
        })
        self.test_df = pl.DataFrame({
            'text': ['text6', 'text7'],
            'id': [6, 7]
        })
    
    @patch('lib.pipeline.stage_pipeline.DataLoader')
    @patch('lib.pipeline.stage_pipeline.Path')
    def test_init(self, mock_path, mock_loader):
        """Test StagePipeline.__init__."""
        mock_loader_instance = Mock()
        mock_loader.return_value = mock_loader_instance
        
        pipeline = StagePipeline(self.config)
        
        self.assertIsNotNone(pipeline.config)
        self.assertIsNotNone(pipeline.logger)
        self.assertIsNotNone(pipeline.data_loader)
    
    @patch('lib.pipeline.stage_pipeline.ExploratoryAnalyzer')
    @patch('lib.pipeline.stage_pipeline.Path')
    def test_stage_1_exploratory(self, mock_path, mock_analyzer):
        """Test stage_1_exploratory method."""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.analyze.return_value = {'result': 'test'}
        mock_analyzer.return_value = mock_analyzer_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.exploratory_analyzer = mock_analyzer_instance
        
        # Mock Path operations
        mock_output_dir = Mock()
        mock_path.return_value.__truediv__.return_value = mock_output_dir
        mock_output_dir.mkdir.return_value = None
        mock_output_dir.__truediv__.return_value.open.return_value.__enter__.return_value = Mock()
        
        result = pipeline.stage_1_exploratory(self.train_df, self.val_df, self.test_df)
        
        self.assertIsInstance(result, dict)
        mock_analyzer_instance.analyze.assert_called_once()
    
    @patch('lib.pipeline.stage_pipeline.StatisticalAnalyzer')
    @patch('lib.pipeline.stage_pipeline.Path')
    def test_stage_2_statistical_analysis(self, mock_path, mock_analyzer):
        """Test stage_2_statistical_analysis method."""
        mock_analyzer_instance = Mock()
        mock_analyzer_instance.comprehensive_statistical_analysis.return_value = {'result': 'test'}
        mock_analyzer.return_value = mock_analyzer_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.stats_analyzer = mock_analyzer_instance
        
        result = pipeline.stage_2_statistical_analysis(self.train_df, self.val_df)
        
        self.assertIsInstance(result, dict)
        mock_analyzer_instance.comprehensive_statistical_analysis.assert_called_once()
    
    @patch('lib.pipeline.stage_pipeline.NLPFeatureExtractor')
    @patch('lib.pipeline.stage_pipeline.EmbeddingExtractor')
    @patch('lib.pipeline.stage_pipeline.FeatureUnion')
    @patch('lib.pipeline.stage_pipeline.ArrowStorage')
    @patch('lib.pipeline.stage_pipeline.collect_after_chunk')
    @patch('lib.pipeline.stage_pipeline.collect_after_operation')
    def test_stage_3_feature_engineering(self, mock_collect_op, mock_collect_chunk, 
                                         mock_arrow, mock_union, mock_emb, mock_nlp):
        """Test stage_3_feature_engineering method."""
        # Mock NLP extractor
        mock_nlp_instance = Mock()
        mock_nlp_instance.extract_all_features.return_value = csr_matrix((3, 100))
        mock_nlp_instance.transform.return_value = csr_matrix((2, 100))
        mock_nlp.return_value = mock_nlp_instance
        
        # Mock embedding extractor
        mock_emb_instance = Mock()
        mock_emb_instance.initialize_embeddings.return_value = None
        mock_emb_instance.extract_all_embeddings.return_value = {}
        mock_emb.return_value = mock_emb_instance
        
        # Mock feature union
        mock_union_instance = Mock()
        mock_union_instance.combine_features.return_value = csr_matrix((3, 100))
        mock_union.return_value = mock_union_instance
        
        # Mock arrow storage
        mock_arrow_instance = Mock()
        mock_arrow.return_value = mock_arrow_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.nlp_extractor = mock_nlp_instance
        pipeline.embedding_extractor = mock_emb_instance
        pipeline.feature_union = mock_union_instance
        pipeline.arrow_storage = mock_arrow_instance
        pipeline.config.use_embeddings = False
        
        result = pipeline.stage_3_feature_engineering(self.train_df, self.val_df, self.test_df)
        
        self.assertIsInstance(result, dict)
        self.assertIn('components', result)
        self.assertIn('nlp', result['components'])
        self.assertIn('train', result['components']['nlp'])
        self.assertIn('val', result['components']['nlp'])
        # Verify GC functions were called
        self.assertTrue(mock_collect_chunk.called or mock_collect_op.called)
    
    @patch('lib.pipeline.stage_pipeline.PreprocessingPipeline')
    @patch('lib.pipeline.stage_pipeline.ArrowStorage')
    @patch('lib.pipeline.stage_pipeline.collect_after_operation')
    def test_stage_4_preprocessing(self, mock_collect, mock_arrow, mock_preproc):
        """Test stage_4_preprocessing method."""
        mock_preproc_instance = Mock()
        mock_preproc_instance.fit_transform.return_value = csr_matrix((3, 100))
        mock_preproc_instance.transform.return_value = csr_matrix((2, 100))
        mock_preproc.return_value = mock_preproc_instance
        
        mock_arrow_instance = Mock()
        mock_arrow.return_value = mock_arrow_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.arrow_storage = mock_arrow_instance
        
        features = {
            'train': csr_matrix((3, 100)),
            'val': csr_matrix((2, 100)),
            'test': csr_matrix((2, 100))
        }
        
        result = pipeline.stage_4_preprocessing(features)
        
        self.assertIsInstance(result, dict)
        self.assertIn('train', result)
        self.assertIn('val', result)
        # Verify GC function was called
        self.assertTrue(mock_collect.called)
    
    @patch('lib.pipeline.stage_pipeline.RecursiveFeatureElimination')
    @patch('lib.pipeline.stage_pipeline.ArrowStorage')
    @patch('lib.pipeline.stage_pipeline.Visualizer')
    @patch('lib.pipeline.stage_pipeline.collect_after_operation')
    def test_stage_5_rfe(self, mock_collect, mock_viz, mock_arrow, mock_rfe):
        """Test stage_5_rfe method."""
        mock_rfe_instance = Mock()
        mock_rfe_instance.fit_transform.return_value = {'n_features_selected': 50}
        mock_rfe_instance.transform.return_value = csr_matrix((3, 50))
        mock_rfe.return_value = mock_rfe_instance
        
        mock_arrow_instance = Mock()
        mock_arrow.return_value = mock_arrow_instance
        
        mock_viz_instance = Mock()
        mock_viz.return_value = mock_viz_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.rfe = mock_rfe_instance
        pipeline.arrow_storage = mock_arrow_instance
        pipeline.visualizer = mock_viz_instance
        pipeline.preprocessing_cache = {}
        
        X_train = csr_matrix((100, 200))
        y_train = np.array([0, 1] * 50)
        X_val = csr_matrix((50, 200))
        y_val = np.array([0, 1] * 25)
        
        result = pipeline.stage_5_rfe(X_train, y_train, X_val, y_val)
        
        self.assertIsInstance(result, dict)
        self.assertIn('train', result)
        self.assertIn('val', result)
        # Verify GC function was called
        self.assertTrue(mock_collect.called)
    
    @patch('lib.pipeline.stage_pipeline.CrossValidator')
    @patch('lib.pipeline.stage_pipeline.GridSearch')
    @patch('lib.pipeline.stage_pipeline.ArrowStorage')
    @patch('lib.pipeline.stage_pipeline.Visualizer')
    def test_stage_6_training_30pct(self, mock_viz, mock_arrow, mock_grid, mock_cv):
        """Test stage_6_training_30pct method."""
        mock_cv_instance = Mock()
        mock_cv_instance.evaluate_model.return_value = {
            'metrics': {'f1': {'mean': 0.8}}
        }
        mock_cv.return_value = mock_cv_instance
        
        mock_grid_instance = Mock()
        mock_grid_instance.search.return_value = {'best_params': {}}
        mock_grid.return_value = mock_grid_instance
        
        mock_arrow_instance = Mock()
        mock_arrow.return_value = mock_arrow_instance
        
        mock_viz_instance = Mock()
        mock_viz.return_value = mock_viz_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.cv = mock_cv_instance
        pipeline.grid_search = mock_grid_instance
        pipeline.arrow_storage = mock_arrow_instance
        pipeline.visualizer = mock_viz_instance
        pipeline.stats_analyzer = None
        pipeline.original_data_size = 1000
        
        X_train = csr_matrix((100, 200))
        y_train = np.array([0, 1] * 50)
        X_val = csr_matrix((50, 200))
        y_val = np.array([0, 1] * 25)
        
        result = pipeline.stage_6_training_30pct(X_train, y_train, X_val, y_val)
        
        self.assertIsInstance(result, dict)
    
    @patch('lib.pipeline.stage_pipeline.CrossValidator')
    @patch('lib.pipeline.stage_pipeline.ArrowStorage')
    @patch('lib.pipeline.stage_pipeline.Visualizer')
    @patch('lib.pipeline.stage_pipeline.collect_after_operation')
    def test_stage_7_full_training(self, mock_collect, mock_viz, mock_arrow, mock_cv):
        """Test stage_7_full_training method."""
        mock_cv_instance = Mock()
        mock_cv_instance.evaluate_model.return_value = {
            'metrics': {'f1': {'mean': 0.8}}
        }
        mock_cv.return_value = mock_cv_instance
        
        mock_arrow_instance = Mock()
        mock_arrow.return_value = mock_arrow_instance
        
        mock_viz_instance = Mock()
        mock_viz.return_value = mock_viz_instance
        
        # Mock model
        mock_model = Mock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.array([0, 1, 0])
        mock_model.predict_proba.return_value = np.array([[0.3, 0.7], [0.8, 0.2], [0.4, 0.6]])
        
        pipeline = StagePipeline(self.config)
        pipeline.cv = mock_cv_instance
        pipeline.arrow_storage = mock_arrow_instance
        pipeline.visualizer = mock_viz_instance
        pipeline.checkpoint_manager = Mock()
        pipeline.checkpoint_manager.save_checkpoint.return_value = "/tmp/model.pkl"
        pipeline.original_data_size = 1000
        
        # Mock model factory
        pipeline._get_model_factory = Mock(return_value=lambda **kwargs: mock_model)
        
        X_train = csr_matrix((100, 200))
        y_train = np.array([0, 1] * 50)
        X_val = csr_matrix((50, 200))
        y_val = np.array([0, 1] * 25)
        
        result = pipeline.stage_7_full_training(
            X_train, y_train, X_val, y_val, 'logreg', {}
        )
        
        self.assertIsInstance(result, dict)
        self.assertIn('model_type', result)
        # Verify GC function was called
        self.assertTrue(mock_collect.called)
    
    def test_get_model_factory(self):
        """Test _get_model_factory method."""
        pipeline = StagePipeline(self.config)
        
        factory = pipeline._get_model_factory('logreg')
        self.assertTrue(callable(factory))
        
        with self.assertRaises(ValueError):
            pipeline._get_model_factory('unknown_model')
    
    @patch('lib.pipeline.stage_pipeline.Path')
    @patch('lib.pipeline.stage_pipeline.shutil')
    def test_cleanup_existing_checkpoints(self, mock_shutil, mock_path):
        """Test _cleanup_existing_checkpoints method."""
        mock_path_instance = Mock()
        mock_path_instance.exists.return_value = True
        mock_path.return_value = mock_path_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.config.delete_existing = True
        
        pipeline._cleanup_existing_checkpoints()
        
        # Should not raise any errors
        self.assertTrue(True)
    
    @patch('lib.pipeline.stage_pipeline.Path')
    def test_save_stage_checkpoint(self, mock_path):
        """Test _save_stage_checkpoint method."""
        mock_output_dir = Mock()
        mock_path.return_value.__truediv__.return_value = mock_output_dir
        mock_output_dir.mkdir.return_value = None
        mock_output_dir.__truediv__.return_value.open.return_value.__enter__.return_value = Mock()
        
        pipeline = StagePipeline(self.config)
        pipeline.arrow_storage = Mock()
        
        pipeline._save_stage_checkpoint('stage_1', {'result': 'test'})
        
        # Should not raise any errors
        self.assertTrue(True)
    
    @patch('lib.pipeline.stage_pipeline.DataLoader')
    def test_run_all_stages(self, mock_loader):
        """Test run_all_stages method."""
        mock_loader_instance = Mock()
        mock_loader_instance.load_train_val_test.return_value = (
            self.train_df, self.val_df, self.test_df
        )
        mock_loader.return_value = mock_loader_instance
        
        pipeline = StagePipeline(self.config)
        pipeline.data_loader = mock_loader_instance
        
        # Mock all stage methods to return quickly
        pipeline.stage_1_exploratory = Mock(return_value={})
        pipeline.stage_2_statistical_analysis = Mock(return_value={})
        pipeline.stage_3_feature_engineering = Mock(return_value={
            'train': csr_matrix((3, 100)),
            'val': csr_matrix((2, 100)),
            'test': csr_matrix((2, 100))
        })
        pipeline.stage_4_preprocessing = Mock(return_value={
            'train': csr_matrix((3, 100)),
            'val': csr_matrix((2, 100)),
            'test': csr_matrix((2, 100))
        })
        pipeline.stage_5_rfe = Mock(return_value={
            'train': csr_matrix((3, 50)),
            'val': csr_matrix((2, 50)),
            'test': csr_matrix((2, 50))
        })
        pipeline.stage_6_training_30pct = Mock(return_value={
            'logreg': {'best_params': {}, 'cv_results': {'metrics': {'f1': {'mean': 0.8}}}}
        })
        pipeline.stage_7_full_training = Mock(return_value={'model_type': 'logreg'})
        pipeline.stage_results = {'best_model': 'logreg'}
        pipeline.config.use_rfe = False
        
        result = pipeline.run_all_stages()
        
        self.assertIsInstance(result, dict)
        self.assertIn('exploratory', result)


if __name__ == '__main__':
    unittest.main()
