"""
Comprehensive visualization utilities for all pipeline stages.
"""
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from scipy.sparse import csr_matrix

try:
    from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, roc_auc_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from ..config import get_config
from ..logging.logger import get_logger

logger = get_logger(__name__)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


class Visualizer:
    """Comprehensive visualization generator."""
    
    def __init__(self, config=None, output_dir: Optional[str] = None):
        """Initialize visualizer."""
        self.config = config or get_config()
        self.logger = get_logger(self.__class__.__name__)
        self.output_dir = Path(output_dir) if output_dir else Path(self.config.output_dir) / "plots"
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_label_distribution(self, labels: np.ndarray, save_path: Optional[str] = None):
        """Plot label distribution."""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Count plot
        unique, counts = np.unique(labels, return_counts=True)
        axes[0].bar(unique, counts, alpha=0.7, color=['skyblue', 'salmon'])
        axes[0].set_xlabel('Label')
        axes[0].set_ylabel('Count')
        axes[0].set_title('Label Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Pie chart
        axes[1].pie(counts, labels=[f'Class {u}' for u in unique], autopct='%1.1f%%', startangle=90)
        axes[1].set_title('Label Distribution (Pie)')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_text_length_distribution(self, lengths: List[int], save_path: Optional[str] = None):
        """Plot text length distribution."""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        lengths = np.array(lengths)
        
        # Histogram
        axes[0, 0].hist(lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[0, 0].set_xlabel('Text Length (characters)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Text Length Distribution')
        axes[0, 0].axvline(np.mean(lengths), color='r', linestyle='--', label=f'Mean: {np.mean(lengths):.1f}')
        axes[0, 0].axvline(np.median(lengths), color='g', linestyle='--', label=f'Median: {np.median(lengths):.1f}')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Box plot
        axes[0, 1].boxplot(lengths, vert=True)
        axes[0, 1].set_ylabel('Text Length')
        axes[0, 1].set_title('Text Length Box Plot')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Log scale histogram
        axes[1, 0].hist(lengths, bins=50, alpha=0.7, edgecolor='black')
        axes[1, 0].set_xlabel('Text Length (log scale)')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].set_title('Text Length Distribution (Log Scale)')
        axes[1, 0].set_xscale('log')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Statistics
        stats_text = f"""Statistics:
Mean: {np.mean(lengths):.2f}
Median: {np.median(lengths):.2f}
Std: {np.std(lengths):.2f}
Min: {np.min(lengths)}
Max: {np.max(lengths)}
Q1: {np.percentile(lengths, 25):.2f}
Q3: {np.percentile(lengths, 75):.2f}"""
        axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_cv_metrics(self, cv_results: Dict[str, Dict[str, Any]], save_path: Optional[str] = None):
        """Plot CV metrics across folds."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Cross-Validation Metrics Across Folds', fontsize=16)
        
        metrics_to_plot = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
        
        for idx, metric in enumerate(metrics_to_plot):
            row = idx // 3
            col = idx % 3
            
            model_names = []
            fold_means = []
            fold_stds = []
            
            for model_name, results in cv_results.items():
                if 'error' not in results and 'metrics' in results:
                    metric_data = results['metrics'].get(metric, {})
                    if 'values' in metric_data:
                        model_names.append(model_name)
                        fold_means.append(metric_data.get('mean', 0))
                        fold_stds.append(metric_data.get('std', 0))
            
            if model_names:
                x_pos = np.arange(len(model_names))
                axes[row, col].bar(x_pos, fold_means, yerr=fold_stds, alpha=0.7, capsize=5)
                axes[row, col].set_xlabel('Model')
                axes[row, col].set_ylabel(metric.upper())
                axes[row, col].set_title(f'{metric.upper()} Across Models')
                axes[row, col].set_xticks(x_pos)
                axes[row, col].set_xticklabels(model_names, rotation=45, ha='right')
                axes[row, col].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_model_comparison(self, results: Dict[str, Dict[str, float]], save_path: Optional[str] = None):
        """Plot model comparison across metrics."""
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Model Comparison', fontsize=16)
        
        metrics = ['f1', 'accuracy', 'precision', 'recall', 'roc_auc']
        
        for idx, metric in enumerate(metrics):
            row = idx // 3
            col = idx % 3
            
            model_names = []
            values = []
            
            for model_name, metrics_dict in results.items():
                if metric in metrics_dict:
                    model_names.append(model_name)
                    values.append(metrics_dict[metric])
            
            if model_names:
                bars = axes[row, col].bar(model_names, values, alpha=0.7)
                axes[row, col].set_ylabel(metric.upper())
                axes[row, col].set_title(f'{metric.upper()} Comparison')
                axes[row, col].set_xticklabels(model_names, rotation=45, ha='right')
                axes[row, col].grid(True, alpha=0.3)
                
                # Add value labels
                for bar, val in zip(bars, values):
                    height = bar.get_height()
                    axes[row, col].text(bar.get_x() + bar.get_width()/2., height,
                                       f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, 
                             model_name: str = "Model", save_path: Optional[str] = None):
        """Plot confusion matrix."""
        if not SKLEARN_AVAILABLE:
            return None
        
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar_kws={'label': 'Count'})
        ax.set_xlabel('Predicted')
        ax.set_ylabel('Actual')
        ax.set_title(f'Confusion Matrix - {model_name}')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_roc_curve(self, y_true: np.ndarray, y_proba: np.ndarray, 
                      model_name: str = "Model", save_path: Optional[str] = None):
        """Plot ROC curve."""
        if not SKLEARN_AVAILABLE or y_proba is None:
            return None
        
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        auc = roc_auc_score(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, label=f'ROC (AUC = {auc:.4f})')
        ax.plot([0, 1], [0, 1], 'k--', label='Random')
        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'ROC Curve - {model_name}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_precision_recall_curve(self, y_true: np.ndarray, y_proba: np.ndarray,
                                   model_name: str = "Model", save_path: Optional[str] = None):
        """Plot Precision-Recall curve."""
        if not SKLEARN_AVAILABLE or y_proba is None:
            return None
        
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(recall, precision)
        ax.set_xlabel('Recall')
        ax.set_ylabel('Precision')
        ax.set_title(f'Precision-Recall Curve - {model_name}')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_feature_importance(self, importances: np.ndarray, feature_names: Optional[List[str]] = None,
                               top_n: int = 20, save_path: Optional[str] = None):
        """Plot feature importance."""
        # Get top N features
        top_indices = np.argsort(importances)[-top_n:][::-1]
        top_importances = importances[top_indices]
        
        if feature_names:
            top_names = [feature_names[i] if i < len(feature_names) else f'Feature_{i}' 
                        for i in top_indices]
        else:
            top_names = [f'Feature_{i}' for i in top_indices]
        
        fig, ax = plt.subplots(figsize=(12, 8))
        bars = ax.barh(range(len(top_names)), top_importances, alpha=0.7)
        ax.set_yticks(range(len(top_names)))
        ax.set_yticklabels(top_names)
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importances')
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_rfe_results(self, rfe_results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot RFE results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Number of features selected
        n_selected = rfe_results.get('n_features_selected', 0)
        n_original = rfe_results.get('n_features_original', 0)
        
        axes[0].bar(['Original', 'Selected'], [n_original, n_selected], 
                   color=['skyblue', 'salmon'], alpha=0.7)
        axes[0].set_ylabel('Number of Features')
        axes[0].set_title('Feature Selection Results')
        axes[0].grid(True, alpha=0.3, axis='y')
        
        # CV scores if available
        if 'cv_scores' in rfe_results and rfe_results['cv_scores']:
            cv_scores = rfe_results['cv_scores']
            n_features_range = np.linspace(n_original, n_selected, len(cv_scores))
            axes[1].plot(n_features_range, cv_scores, 'o-')
            axes[1].set_xlabel('Number of Features')
            axes[1].set_ylabel('CV Score')
            axes[1].set_title('RFE CV Scores')
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_anova_results(self, anova_results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot ANOVA results."""
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # F-statistic and p-value
        f_stat = anova_results.get('f_statistic', 0)
        p_value = anova_results.get('p_value', 1)
        
        axes[0].bar(['F-statistic', 'p-value'], [f_stat, p_value], 
                   color=['steelblue', 'coral'], alpha=0.7)
        axes[0].set_ylabel('Value')
        axes[0].set_title('ANOVA Test Statistics')
        axes[0].grid(True, alpha=0.3, axis='y')
        axes[0].axhline(0.05, color='r', linestyle='--', label='α=0.05')
        axes[0].legend()
        
        # Group means
        group_means = anova_results.get('group_means', {})
        if group_means:
            groups = list(group_means.keys())
            means = list(group_means.values())
            bars = axes[1].bar(groups, means, alpha=0.7)
            axes[1].set_ylabel('Mean Metric Value')
            axes[1].set_xlabel('Model')
            axes[1].set_title('Group Means')
            axes[1].set_xticklabels(groups, rotation=45, ha='right')
            axes[1].grid(True, alpha=0.3, axis='y')
            
            # Add value labels
            for bar, val in zip(bars, means):
                height = bar.get_height()
                axes[1].text(bar.get_x() + bar.get_width()/2., height,
                           f'{val:.4f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_tukey_hsd(self, tukey_results: Dict[str, Any], save_path: Optional[str] = None):
        """Plot Tukey HSD results."""
        # This would require the full Tukey results table
        # For now, create a summary plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Extract significant pairs if available
        significant_pairs = tukey_results.get('significant_pairs', [])
        
        if significant_pairs:
            pairs = [f"{p[0]} vs {p[1]}" for p in significant_pairs]
            ax.text(0.5, 0.5, f"Significant pairs: {len(significant_pairs)}\n" + 
                   "\n".join(pairs[:10]), 
                   ha='center', va='center', fontsize=12,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        else:
            ax.text(0.5, 0.5, "No significant differences found", 
                   ha='center', va='center', fontsize=12)
        
        ax.axis('off')
        ax.set_title('Tukey HSD Post-hoc Test Results')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig
    
    def plot_training_curves(self, train_metrics: Dict[str, List[float]], 
                           val_metrics: Dict[str, List[float]],
                           save_path: Optional[str] = None):
        """Plot training curves."""
        n_epochs = len(train_metrics.get('loss', []))
        if n_epochs == 0:
            return None
        
        epochs = range(1, n_epochs + 1)
        n_metrics = len([k for k in train_metrics.keys() if k != 'loss'])
        
        fig, axes = plt.subplots(2, max(2, (n_metrics + 1) // 2), figsize=(16, 10))
        axes = axes.flatten()
        fig.suptitle('Training Progress', fontsize=16)
        
        idx = 0
        
        # Loss
        if 'loss' in train_metrics:
            axes[idx].plot(epochs, train_metrics['loss'], 'b-', label='Train Loss')
            if 'loss' in val_metrics:
                axes[idx].plot(epochs, val_metrics['loss'], 'r-', label='Val Loss')
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].set_title('Loss')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
            idx += 1
        
        # Other metrics
        for metric in ['f1', 'accuracy', 'precision', 'recall']:
            if metric in train_metrics and idx < len(axes):
                axes[idx].plot(epochs, train_metrics[metric], 'b-', label=f'Train {metric}')
                if metric in val_metrics:
                    axes[idx].plot(epochs, val_metrics[metric], 'r-', label=f'Val {metric}')
                axes[idx].set_xlabel('Epoch')
                axes[idx].set_ylabel(metric.upper())
                axes[idx].set_title(metric.upper())
                axes[idx].legend()
                axes[idx].grid(True, alpha=0.3)
                idx += 1
        
        # Hide unused subplots
        for i in range(idx, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        return fig

