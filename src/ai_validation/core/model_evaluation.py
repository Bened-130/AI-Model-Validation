"""
Model Evaluation Framework
Standardized evaluation with statistical testing and cross-validation
"""

import numpy as np
import pandas as pd
from scipy import stats
from typing import Dict, Any, List, Optional
from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, confusion_matrix
)
import logging

logger = logging.getLogger(__name__)


class ModelEvaluationFramework:
    """Standardized framework for model evaluation"""
    
    def __init__(self, framework_version: str = 'v3.0'):
        self.version = framework_version
        self.evaluation_history: List[Dict] = []
        logger.info(f"ModelEvaluationFramework {framework_version} initialized")
    
    def evaluate_classifier(
        self, 
        model, 
        X_test: np.ndarray, 
        y_test: np.ndarray,
        model_name: str = "Model"
    ) -> Dict[str, Any]:
        """Evaluate classifier and return comprehensive metrics"""
        logger.info(f"Evaluating {model_name}")
        
        y_pred = model.predict(X_test)
        
        # Handle predict_proba for ROC-AUC
        y_pred_proba = None
        if hasattr(model, 'predict_proba'):
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
            except Exception:
                pass
        
        metrics = {
            'model_name': model_name,
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
            'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
            'f1_score': f1_score(y_test, y_pred, average='weighted', zero_division=0),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'n_samples': len(y_test)
        }
        
        if y_pred_proba is not None:
            try:
                metrics['roc_auc'] = roc_auc_score(y_test, y_pred_proba)
            except ValueError:
                metrics['roc_auc'] = None
        
        # Calculate improvement vs baseline
        if len(self.evaluation_history) > 0:
            baseline_acc = self.evaluation_history[0]['accuracy']
            improvement = ((metrics['accuracy'] - baseline_acc) / baseline_acc) * 100
            metrics['improvement_vs_baseline'] = round(improvement, 2)
        
        self.evaluation_history.append(metrics)
        logger.info(f"{model_name} accuracy: {metrics['accuracy']:.4f}")
        
        return metrics
    
    def cross_validate_model(
        self, 
        model, 
        X: np.ndarray, 
        y: np.ndarray, 
        cv: int = 5
    ) -> Dict[str, Any]:
        """Perform k-fold cross-validation"""
        logger.info(f"Running {cv}-fold cross-validation")
        
        scores = cross_val_score(model, X, y, cv=cv, scoring='accuracy', n_jobs=-1)
        
        return {
            'cv_scores': scores.tolist(),
            'mean_score': float(scores.mean()),
            'std_score': float(scores.std()),
            'cv_folds': cv
        }
    
    def compare_models(self, evaluations: List[Dict]) -> pd.DataFrame:
        """Create comparison table from evaluation results"""
        rows = []
        for eval_result in evaluations:
            rows.append({
                'Model': eval_result['model_name'],
                'Accuracy': f"{eval_result['accuracy']:.4f}",
                'Precision': f"{eval_result['precision']:.4f}",
                'Recall': f"{eval_result['recall']:.4f}",
                'F1 Score': f"{eval_result['f1_score']:.4f}",
                'Samples': eval_result['n_samples']
            })
        
        return pd.DataFrame(rows)
    
    def statistical_significance_test(
        self,
        model1_scores: List[float],
        model2_scores: List[float]
    ) -> Dict[str, Any]:
        """Test if difference between models is statistically significant"""
        logger.info("Running t-test for statistical significance")
        
        t_stat, p_value = stats.ttest_ind(model1_scores, model2_scores)
        
        # Cohen's d effect size
        mean1, mean2 = np.mean(model1_scores), np.mean(model2_scores)
        std1, std2 = np.std(model1_scores, ddof=1), np.std(model2_scores, ddof=1)
        n1, n2 = len(model1_scores), len(model2_scores)
        
        pooled_std = np.sqrt(((n1-1)*std1**2 + (n2-1)*std2**2) / (n1+n2-2))
        cohens_d = (mean1 - mean2) / pooled_std if pooled_std > 0 else 0
        
        return {
            't_statistic': float(t_stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'cohens_d': float(cohens_d),
            'effect_size': self._interpret_effect_size(cohens_d)
        }
    
    def _interpret_effect_size(self, d: float) -> str:
        """Interpret Cohen's d effect size"""
        abs_d = abs(d)
        if abs_d < 0.2:
            return "Small"
        elif abs_d < 0.5:
            return "Medium"
        else:
            return "Large"