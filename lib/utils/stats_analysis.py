"""
Statistical analyses: ANOVA, Tukey HSD, UMAP, etc.
"""
import numpy as np
from typing import Dict, List, Any, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

try:
    from scipy import stats
    from scipy.stats import f_oneway
    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)


class StatisticalAnalyzer:
    """Perform statistical analyses on model results."""
    
    def __init__(self, config=None):
        """
        Initialize statistical analyzer.
        
        Args:
            config: Configuration object
        """
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
    
    def anova_test(
        self,
        groups: Dict[str, np.ndarray],
        metric: str = "f1"
    ) -> Dict[str, Any]:
        """
        Perform ANOVA test on model groups.
        
        Args:
            groups: Dictionary of model_name -> metric_values array
            metric: Name of metric being tested
            
        Returns:
            Dictionary with ANOVA results
        """
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy not available, skipping ANOVA")
            return {}
        
        self.logger.info(f"Performing ANOVA test on {metric}")
        
        group_names = list(groups.keys())
        group_data = [groups[name] for name in group_names]
        
        # Perform one-way ANOVA
        f_stat, p_value = f_oneway(*group_data)
        
        results = {
            'f_statistic': float(f_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'groups': group_names,
            'group_means': {name: float(np.mean(groups[name])) for name in group_names},
            'group_stds': {name: float(np.std(groups[name])) for name in group_names}
        }
        
        self.logger.info(
            f"ANOVA Results - F={f_stat:.4f}, p={p_value:.4f}, "
            f"Significant: {results['significant']}"
        )
        
        return results
    
    def tukey_hsd(
        self,
        groups: Dict[str, np.ndarray],
        metric: str = "f1"
    ) -> Dict[str, Any]:
        """
        Perform Tukey HSD post-hoc test.
        
        Args:
            groups: Dictionary of model_name -> metric_values array
            metric: Name of metric being tested
            
        Returns:
            Dictionary with Tukey HSD results
        """
        if not SCIPY_AVAILABLE:
            self.logger.warning("scipy/statsmodels not available, skipping Tukey HSD")
            return {}
        
        self.logger.info(f"Performing Tukey HSD test on {metric}")
        
        # Prepare data for Tukey HSD
        data = []
        group_labels = []
        
        for name, values in groups.items():
            data.extend(values)
            group_labels.extend([name] * len(values))
        
        # Perform Tukey HSD
        tukey_result = pairwise_tukeyhsd(data, group_labels, alpha=0.05)
        
        results = {
            'summary': str(tukey_result),
            'reject': tukey_result.reject.tolist(),
            'meandiff': tukey_result.meandiff.tolist(),
            'p_adj': tukey_result.p_adj.tolist(),
            'groups': tukey_result.groups_unique.tolist()
        }
        
        self.logger.info("Tukey HSD completed")
        return results
    
    def umap_visualization(
        self,
        X: np.ndarray,
        y: Optional[np.ndarray] = None,
        n_components: int = 2,
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        save_path: Optional[str] = None
    ) -> np.ndarray:
        """
        Perform UMAP dimensionality reduction and visualization.
        
        Args:
            X: Feature matrix
            y: Optional labels for coloring
            n_components: Number of UMAP dimensions
            n_neighbors: UMAP n_neighbors parameter
            min_dist: UMAP min_dist parameter
            save_path: Path to save visualization
            
        Returns:
            UMAP embedding
        """
        if not UMAP_AVAILABLE:
            self.logger.warning("umap not available, skipping UMAP")
            return X
        
        self.logger.info("Performing UMAP dimensionality reduction")
        
        # Fit UMAP
        reducer = umap.UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=self.config.random_state
        )
        
        embedding = reducer.fit_transform(X)
        
        # Visualize if 2D
        if n_components == 2 and y is not None:
            fig, ax = plt.subplots(figsize=(10, 8))
            scatter = ax.scatter(
                embedding[:, 0],
                embedding[:, 1],
                c=y,
                cmap='viridis',
                alpha=0.6,
                s=20
            )
            ax.set_xlabel('UMAP 1')
            ax.set_ylabel('UMAP 2')
            ax.set_title('UMAP Visualization')
            plt.colorbar(scatter, ax=ax, label='Label')
            plt.tight_layout()
            
            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            
            plt.close(fig)
        
        self.logger.info(f"UMAP completed: {X.shape} -> {embedding.shape}")
        return embedding
    
    def plot_metric_comparison(
        self,
        results: Dict[str, Dict[str, float]],
        metric: str = "f1",
        save_path: Optional[str] = None
    ):
        """Plot metric comparison across models."""
        model_names = list(results.keys())
        metric_values = [results[name].get(metric, 0.0) for name in model_names]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(model_names, metric_values, alpha=0.7)
        ax.set_ylabel(metric.upper())
        ax.set_xlabel('Model')
        ax.set_title(f'{metric.upper()} Comparison Across Models')
        ax.grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, val in zip(bars, metric_values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{val:.4f}',
                   ha='center', va='bottom')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.close(fig)
        return fig
    
    def comprehensive_analysis(
        self,
        cv_results: Dict[str, Dict[str, List[float]]],
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Perform comprehensive statistical analysis.
        
        Args:
            cv_results: Dictionary of model_name -> metrics dict
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with all analysis results
        """
        self.logger.info("Performing comprehensive statistical analysis")
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        analysis_results = {}
        
        # Extract F1 scores for each model
        f1_groups = {}
        for model_name, metrics in cv_results.items():
            if 'f1' in metrics and 'values' in metrics['f1']:
                f1_groups[model_name] = np.array(metrics['f1']['values'])
        
        if len(f1_groups) > 1:
            # ANOVA
            anova_results = self.anova_test(f1_groups, metric="f1")
            analysis_results['anova'] = anova_results
            
            # Tukey HSD if ANOVA is significant
            if anova_results.get('significant', False):
                tukey_results = self.tukey_hsd(f1_groups, metric="f1")
                analysis_results['tukey_hsd'] = tukey_results
            
            # Plot comparison
            model_means = {name: np.mean(values) for name, values in f1_groups.items()}
            if save_dir:
                self.plot_metric_comparison(
                    {name: {'f1': mean} for name, mean in model_means.items()},
                    metric="f1",
                    save_path=str(Path(save_dir) / "f1_comparison.png")
                )
        
        self.logger.info("Comprehensive analysis completed")
        return analysis_results
    
    def comprehensive_statistical_analysis(
        self,
        train_df,
        val_df,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive statistical analysis including ANOVA, Tukey HSD, F statistic, Cramer's V.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with all statistical results
        """
        self.logger.info("Performing comprehensive statistical analysis")
        results = {}
        
        if save_dir:
            Path(save_dir).mkdir(parents=True, exist_ok=True)
        
        # Cramer's V for categorical associations
        if self.config.label_column in train_df.columns:
            results['cramers_v'] = self.cramers_v_analysis(train_df, val_df, save_dir)
        
        # F statistic for feature importance
        if self.config.label_column in train_df.columns:
            results['f_statistic'] = self.f_statistic_analysis(train_df, save_dir)
        
        # ANOVA and Tukey HSD (if we have model results)
        # This will be called separately with model results
        
        return results
    
    def cramers_v_analysis(
        self,
        train_df,
        val_df,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate Cramer's V for categorical feature associations.
        
        Args:
            train_df: Training DataFrame
            val_df: Validation DataFrame
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with Cramer's V results
        """
        if not SCIPY_AVAILABLE:
            return {}
        
        self.logger.info("Calculating Cramer's V")
        
        # For text data, we'd need to extract categorical features
        # For now, return placeholder
        results = {
            'note': 'Cramer\'s V requires categorical features. Text features need encoding first.'
        }
        
        return results
    
    def f_statistic_analysis(
        self,
        train_df,
        save_dir: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Calculate F-statistic for feature importance.
        
        Args:
            train_df: Training DataFrame
            save_dir: Directory to save plots
            
        Returns:
            Dictionary with F-statistic results
        """
        if not SCIPY_AVAILABLE:
            return {}
        
        self.logger.info("Calculating F-statistics")
        
        # F-statistic requires numerical features
        # This would be calculated after feature engineering
        results = {
            'note': 'F-statistic calculated after feature engineering'
        }
        
        return results

